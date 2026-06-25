import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from ..evaluation import eval_standard
from ..utils import precompute_train_embeddings, make_cached_loader
from .loss import contrastive_loss


# ----------------- Forward-pass dispatcher -----------------
def _forward_pass(model, batch, model_type, device):
    if model_type in ['lstm', 'gru']:
        images, captions, lengths, image_names = batch
        images   = images.to(device, non_blocking=True)
        captions = captions.to(device, non_blocking=True)
        img_emb, txt_emb = model(images, captions, lengths, None)

    elif model_type == 'minilm':
        images, input_ids, lengths, image_names, attention_mask = batch
        images         = images.to(device, non_blocking=True)
        input_ids      = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        img_emb, txt_emb = model(images, input_ids, lengths, attention_mask)

    else:
        raise ValueError(f"model_type '{model_type}' không hợp lệ!")

    return img_emb, txt_emb, image_names


# ----------------- Helpers -----------------
def _supports_cache(model_type):
    return model_type in ['lstm', 'gru', 'minilm']


def _is_backbone_frozen(model, model_type):
    """Check cả image backbone và text backbone (với minilm)"""
    img_frozen = not any(
        p.requires_grad for p in model.image_encoder.backbone.parameters()
    )
    if model_type == 'minilm':
        txt_frozen = not any(
            p.requires_grad for p in model.text_encoder.backbone.parameters()
        )
        return img_frozen and txt_frozen
    return img_frozen


# ----------------- device_type helper -----------------
def _device_type(device):
    """Trả về 'cuda' hoặc 'cpu'"""
    raw = device.type if isinstance(device, torch.device) else str(device)
    return raw.split(':')[0]


# ----------------- Optimizer factory -----------------
def _make_optimizer(model, base_lr, cfg):
    """
    Differential LR: backbone dùng FINETUNE_LR, projection dùng LEARNING_RATE.
    Nếu backbone đang frozen thì chỉ tạo 1 group (projection), add_param_group lúc unfreeze vẫn hoạt động bình thường.
    """
    finetune_lr  = cfg.get('FINETUNE_LR', base_lr * 0.1)
    weight_decay = cfg.get('WEIGHT_DECAY', 1e-2)

    backbone_ids    = {id(p) for p in model.image_encoder.backbone.parameters()}
    backbone_params = [p for p in model.image_encoder.backbone.parameters()
                       if p.requires_grad]
    other_params    = [p for p in model.parameters()
                       if id(p) not in backbone_ids and p.requires_grad]

    param_groups = [{'params': other_params, 'lr': base_lr}]
    if backbone_params:
        param_groups.append({'params': backbone_params, 'lr': finetune_lr})

    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


# ----------------- BatchNorm freeze helper -----------------
def _set_bn_eval(model):
    for mod in model.image_encoder.backbone.modules():
        if isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm1d)):
            mod.eval()


# ----------------- Scheduler factory -----------------
def _make_scheduler(optimizer, cfg, warmup_epochs, start_epoch=0):
    """
    Tạo LambdaLR scheduler bắt đầu tại start_epoch trên cosine curve
    """
    total_epochs = cfg['NUM_EPOCHS']

    def lr_lambda(epoch):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))

    last_epoch = max(start_epoch - 1, -1)

    # Đảm bảo mọi param group đều có initial_lr
    for group in optimizer.param_groups:
        if "initial_lr" not in group:
            group["initial_lr"] = group["lr"]

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda, last_epoch=last_epoch
    )


# ----------------- Projection Warmup Phase -----------------
def _run_proj_warmup(model, train_img, train_txt, train_names, unique_names,
                     val_loader, cfg, device, model_type, num_epochs,
                     optimizer, scaler, eval_every=2):
    """
    Phase 1: chỉ train projection heads với precomputed features.
    - Dùng chung optimizer/scaler để giữ moment estimates cho main loop.
    - Scheduler riêng chỉ cho warmup phase, KHÔNG ảnh hưởng scheduler main loop.
    """
    dtype       = _device_type(device)
    temperature = cfg.get('TEMPERATURE', 0.1)
    margin      = cfg.get('MARGIN', 0.2)
    hard_neg_w  = cfg.get('HARD_NEG_WEIGHT', 0.3)

    cached_loader = make_cached_loader(
        train_img, train_txt, train_names, unique_names,
        batch_size=cfg['BATCH_SIZE']
    )

    # Scheduler nội bộ cho proj warmup 
    warmup_sched = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda ep: (ep + 1) / 2 if ep < 2
                   else 0.5 * (1 + np.cos(np.pi * (ep - 2) / max(1, num_epochs - 2)))
    )

    best_r1 = 0.0
    print(f"\n{'='*60}")
    print(f"  PROJECTION WARMUP — {num_epochs} epochs (projection only)")
    print(f"{'='*60}")

    for epoch in range(num_epochs):
        model.train()
        model.image_encoder.backbone.eval()
        if model_type == 'minilm':
            model.text_encoder.backbone.eval()

        train_loss = 0.0
        n_batches  = 0

        for txt_idx, img_idx in tqdm(
                cached_loader,
                desc=f'  Warmup {epoch+1}/{num_epochs}', leave=False):

            img_feat    = train_img[img_idx].to(device, non_blocking=True)
            txt_feat    = train_txt[txt_idx].to(device, non_blocking=True)
            batch_names = [train_names[i] for i in txt_idx.tolist()]

            optimizer.zero_grad()
            with autocast(device_type=dtype, dtype=torch.float16):
                img_emb = F.normalize(model.image_encoder.projection(img_feat), p=2, dim=1)
                txt_emb = F.normalize(model.text_encoder.projection(txt_feat),  p=2, dim=1)
                loss = contrastive_loss(
                    img_emb, txt_emb, batch_names,
                    temperature=temperature, margin=margin, hard_neg_weight=hard_neg_w
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            n_batches  += 1

        warmup_sched.step()

        do_eval = (epoch + 1) % eval_every == 0 or epoch == num_epochs - 1
        if do_eval:
            metrics = eval_standard(model, val_loader, device, model_type)
            r1 = metrics['R@1']
            if r1 > best_r1:
                best_r1 = r1
            print(f"  Warmup epoch {epoch+1:2d}/{num_epochs} | "
                  f"loss {train_loss/n_batches:.4f} | "
                  f"R@1 {r1}%  (best {best_r1}%)")

    print(f"\n  [Proj Warmup done] Best R@1: {best_r1}%")
    print(f"{'='*60}\n")
    return best_r1


# ----------------- Main training -----------------
def train_model(model, train_loader, val_loader, cfg, device, model_type, name_exp='cnn_lstm', save_path='best_model.pth',
                unfreeze_epoch=None, patience=5, eval_every=2, proj_warmup_epochs=0):
    dtype         = _device_type(device)
    writer        = SummaryWriter(f'runs/{name_exp}')
    base_lr       = cfg['LEARNING_RATE']
    warmup_epochs = cfg.get('WARMUP_EPOCHS', 2)
    temperature   = cfg.get('TEMPERATURE', 0.1)
    margin        = cfg.get('MARGIN', 0.2)
    hard_neg_w    = cfg.get('HARD_NEG_WEIGHT', 0.3)

    optimizer = _make_optimizer(model, base_lr, cfg)
    scheduler = _make_scheduler(optimizer, cfg, warmup_epochs, start_epoch=0)
    scaler    = GradScaler(dtype)

    # ----------------- Pre-compute -----------------
    use_cache     = False
    cached_loader = None
    train_img = train_txt = train_names = unique_names = None

    if _supports_cache(model_type) and _is_backbone_frozen(model, model_type):
        print(f"[{model_type}] Backbone đang frozen -> Pre-computing embeddings...")
        train_img, train_txt, train_names, unique_names = precompute_train_embeddings(
            model, train_loader, device, model_type
        )
        cached_loader = make_cached_loader(
            train_img, train_txt, train_names, unique_names,
            batch_size=cfg['BATCH_SIZE']
        )
        use_cache = True

    # ----------------- Projection Warmup -----------------
    if proj_warmup_epochs > 0:
        if not use_cache:
            print("[Warning] proj_warmup_epochs yêu cầu backbone phải frozen, bỏ qua proj warmup!")
        else:
            # Lưu LR trước warmup để restore về đúng điểm ban đầu
            initial_lrs = [g['lr'] for g in optimizer.param_groups]

            _run_proj_warmup(
                model, train_img, train_txt, train_names, unique_names,
                val_loader, cfg, device, model_type,
                num_epochs=proj_warmup_epochs,
                optimizer=optimizer,
                scaler=scaler,
                eval_every=eval_every
            )

            # Restore LR — giữ moment estimates của AdamW, reset LR về base
            for g, lr in zip(optimizer.param_groups, initial_lrs):
                g['lr'] = lr

            scheduler = _make_scheduler(
                optimizer, cfg, warmup_epochs, start_epoch=warmup_epochs
            )

    # ----------------- Main Loop -----------------
    best_r1 = -1.0
    no_improve_epochs = 0                
    history = {'train_loss': [], 'val_loss': [], 'val_r1': [], 'val_r5': [], 'val_r10': []}
    last_metrics = {'R@1': 0.0, 'R@5': 0.0, 'R@10': 0.0}

    print(f"{'='*60}")
    print(f"  MAIN TRAINING — {cfg['NUM_EPOCHS']} epochs  [{model_type}]")
    print(f"  backbone LR : {cfg.get('FINETUNE_LR', base_lr*0.1):.1e}  |  "
          f"projection LR: {base_lr:.1e}")
    print(f"  patience={patience} (tính theo lần eval, mỗi {eval_every} epoch)")
    if unfreeze_epoch is not None:
        print(f"  unfreeze_epoch={unfreeze_epoch}")
    print(f"{'='*60}\n")

    for epoch in range(cfg['NUM_EPOCHS']):

        # ----------------- Unfreeze backbone -----------------
        if unfreeze_epoch is not None and epoch == unfreeze_epoch:
            if _is_backbone_frozen(model, model_type):
                from_layer = cfg.get('UNFREEZE_FROM', 'layer4')
                print(f"\n[Epoch {epoch+1}] [{model_type}] === FINE-TUNE from {from_layer} ===")

                model.image_encoder.unfreeze_from(from_layer)

                optimizer.add_param_group({
                    'params': [p for p in model.image_encoder.backbone.parameters() if p.requires_grad],
                    'lr': cfg.get('FINETUNE_LR', base_lr * 0.1)
                })

                scheduler = _make_scheduler(
                    optimizer, cfg, warmup_epochs=0, start_epoch=epoch
                )

                # Giải phóng cache tường minh trước empty_cache
                del train_img, train_txt, train_names, unique_names
                use_cache     = False
                cached_loader = None
                train_img = train_txt = train_names = unique_names = None
                torch.cuda.empty_cache()
                print(f"    --> {from_layer} unfrozen, chuyển sang full forward pass")

        model.train()

        if _is_backbone_frozen(model, model_type):
            model.image_encoder.backbone.eval()
        else:
            _set_bn_eval(model)

        if model_type == 'minilm':
            model.text_encoder.backbone.eval()

        train_loss = 0.0
        n_batches  = 0

        # ----------------- Train step -----------------
        if use_cache:
            for txt_idx, img_idx in tqdm(
                    cached_loader,
                    desc=f'Epoch {epoch+1}/{cfg["NUM_EPOCHS"]} Train', leave=False):

                img_feat    = train_img[img_idx].to(device, non_blocking=True)
                txt_feat    = train_txt[txt_idx].to(device, non_blocking=True)
                batch_names = [train_names[i] for i in txt_idx.tolist()]

                optimizer.zero_grad()
                with autocast(device_type=dtype, dtype=torch.float16):
                    img_emb = F.normalize(model.image_encoder.projection(img_feat), p=2, dim=1)
                    txt_emb = F.normalize(model.text_encoder.projection(txt_feat),  p=2, dim=1)
                    loss = contrastive_loss(
                        img_emb, txt_emb, batch_names,
                        temperature=temperature, margin=margin, hard_neg_weight=hard_neg_w
                    )

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()
                n_batches  += 1

        else:
            for batch in tqdm(
                    train_loader,
                    desc=f'Epoch {epoch+1}/{cfg["NUM_EPOCHS"]} Train', leave=False):

                optimizer.zero_grad()
                with autocast(device_type=dtype, dtype=torch.float16):
                    img_emb, txt_emb, image_names = _forward_pass(
                        model, batch, model_type, device)
                    loss = contrastive_loss(
                        img_emb, txt_emb, image_names,
                        temperature=temperature, margin=margin, hard_neg_weight=hard_neg_w
                    )

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()
                n_batches  += 1

        # ----------------- Val loss -----------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(
                    val_loader,
                    desc=f'Epoch {epoch+1}/{cfg["NUM_EPOCHS"]} Val', leave=False):

                with autocast(device_type=dtype, dtype=torch.float16):
                    img_emb, txt_emb, image_names = _forward_pass(
                        model, batch, model_type, device)
                    loss = contrastive_loss(
                        img_emb, txt_emb, image_names,
                        temperature=temperature, margin=margin, hard_neg_weight=hard_neg_w
                    )
                val_loss += loss.item()

        # ----------------- Eval & log -----------------
        do_eval = (epoch + 1) % eval_every == 0 or epoch == cfg['NUM_EPOCHS'] - 1
        if do_eval:
            last_metrics = eval_standard(model, val_loader, device, model_type)

        avg_train  = train_loss / n_batches
        avg_val    = val_loss / len(val_loader)
        current_lr = optimizer.param_groups[0]['lr']
        r1, r5, r10 = last_metrics['R@1'], last_metrics['R@5'], last_metrics['R@10']

        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)
        if do_eval:
            history['val_r1'].append(r1)
            history['val_r5'].append(r5)
            history['val_r10'].append(r10)

        eval_tag = '*' if do_eval else ' '
        print(f"Epoch {epoch+1:3d} [{model_type}] | "
              f"Train: {avg_train:.4f} | Val: {avg_val:.4f} | "
              f"R@1: {r1}%{eval_tag} R@5: {r5}% R@10: {r10}% | "
              f"LR: {current_lr:.2e}")

        writer.add_scalar('Loss/train',  avg_train,   epoch)
        writer.add_scalar('Loss/val',    avg_val,     epoch)
        writer.add_scalar('LR',          current_lr,  epoch)
        writer.add_scalar('temperature', temperature, epoch)
        if do_eval:
            writer.add_scalar('Recall/R@1',  r1,  epoch)
            writer.add_scalar('Recall/R@5',  r5,  epoch)
            writer.add_scalar('Recall/R@10', r10, epoch)

        # ----------------- Scheduler step -----------------
        scheduler.step()

        if do_eval:
            if r1 > best_r1:
                best_r1 = r1
                no_improve_epochs = 0
                torch.save(model.state_dict(), save_path)
            else:
                no_improve_epochs += eval_every   # mỗi lần eval = eval_every epoch
                if no_improve_epochs >= patience:
                    print(f'[EarlyStopping] Epoch {epoch+1} | '
                          f'No improvement for {no_improve_epochs} epochs | '
                          f'Best R@1: {best_r1}%')
                    break

    writer.close()

    print(f'\nBest Val R@1: {best_r1}%  (model_type={model_type})')
    print(f'Loading best model from {save_path}...')
    model.load_state_dict(torch.load(save_path, map_location=device))

    return history