import os
import copy
import time
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

from ..utils.embeddings import precompute_train_embeddings, make_cached_loader
from ..evaluation import eval_standard
from .loss import contrastive_loss


#  --------------- Helpers ---------------
def _reset_projection(module):
    for layer in module:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


def _reset_projections(model):
    _reset_projection(model.image_encoder.projection)
    _reset_projection(model.text_encoder.projection)


def _make_proj_optimizer(model, lr, weight_decay=1e-2):
    params = (
        list(model.image_encoder.projection.parameters()) +
        list(model.text_encoder.projection.parameters())
    )
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def _lr_lambda(epoch, num_epochs, warmup=2):
    if epoch < warmup:
        return (epoch + 1) / warmup
    progress = (epoch - warmup) / max(1, num_epochs - warmup)
    return 0.5 * (1 + np.cos(np.pi * progress))


def _make_exp_name(model_type, keys, combo):
    parts = [model_type]
    for k, v in zip(keys, combo):
        short = (k.lower()
                  .replace('learning_rate', 'lr')
                  .replace('temperature', 'temp')
                  .replace('hard_neg_weight', 'hard_neg')
                  .replace('weight_decay', 'wd')
                  .replace('batch_size', 'bs')
                  .replace('margin', 'margin'))
        if isinstance(v, float):
            v_str = f'{v:.0e}' if v < 0.01 else f'{v:.4g}'
        else:
            v_str = str(v)
        parts.append(f'{short}={v_str}')
    return '__'.join(parts)


#  --------------- Single trial ---------------
def _run_trial(model, train_img, train_txt, train_names, unique_names,
               val_loader, cfg, device, model_type, num_epochs,
               writer, exp_name):
    device_type = device.type if isinstance(device, torch.device) else str(device)
    lr          = cfg['LEARNING_RATE']
    temperature = cfg['TEMPERATURE']
    hard_neg_w  = cfg['HARD_NEG_WEIGHT']
    margin      = cfg.get('MARGIN', 0.3)
    batch_size  = cfg.get('BATCH_SIZE', 128)

    _reset_projections(model)

    cached_loader = make_cached_loader(
        train_img, train_txt, train_names, unique_names,
        batch_size=batch_size
    )

    optimizer = _make_proj_optimizer(model, lr, cfg.get('WEIGHT_DECAY', 1e-2))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda ep: _lr_lambda(ep, num_epochs)
    )
    scaler = GradScaler(device_type)

    best_r1 = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        n_batches  = 0

        for txt_idx, img_idx in cached_loader:
            img_feat    = train_img[img_idx].to(device, non_blocking=True)
            txt_feat    = train_txt[txt_idx].to(device, non_blocking=True)
            batch_names = [train_names[i] for i in txt_idx.tolist()]

            optimizer.zero_grad()
            with autocast(device_type=device_type, dtype=torch.float16):
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

        scheduler.step()

        avg_loss   = train_loss / max(n_batches, 1)
        current_lr = optimizer.param_groups[0]['lr']

        writer.add_scalar(f'{exp_name}/loss', avg_loss, epoch)
        writer.add_scalar(f'{exp_name}/lr', current_lr, epoch)

        if (epoch + 1) % 2 == 0 or epoch == num_epochs - 1:
            metrics = eval_standard(model, val_loader, device, model_type)
            r1, r5, r10 = metrics['R@1'], metrics['R@5'], metrics['R@10']

            writer.add_scalar(f'{exp_name}/R@1',  r1,  epoch)
            writer.add_scalar(f'{exp_name}/R@5',  r5,  epoch)
            writer.add_scalar(f'{exp_name}/R@10', r10, epoch)

            if r1 > best_r1:
                best_r1 = r1

        print(f"  Epoch {epoch+1:2d}/{num_epochs} | "
              f"loss {avg_loss:.4f} | best R@1 {best_r1:.1f}%",
              end='\r')

    print()
    return best_r1


#  --------------- Grid search chính ---------------
def grid_search(model, train_loader, val_loader, base_cfg, device, model_type,
                num_epochs=10, param_grid=None, save_dir='./grid_search_results',
                tb_log_dir='runs/grid_search'):
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  GRID SEARCH — model_type: {model_type.upper()}")
    print(f"{'='*55}")
    print(f"[1/2] Pre-computing features ({model_type})...")
    train_img, train_txt, train_names, unique_names = precompute_train_embeddings(
        model, train_loader, device, model_type
    )

    keys   = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(itertools.product(*values))
    total  = len(combos)
    print(f"[2/2] Bắt đầu grid search: {total} tổ hợp × {num_epochs} epochs\n")

    # Writer cho loss/metrics curves theo epoch
    curves_writer = SummaryWriter(log_dir=os.path.join(tb_log_dir, model_type, 'curves'))
    # Writer riêng cho HParams — TensorBoard dùng để so sánh param
    hparam_writer = SummaryWriter(log_dir=os.path.join(tb_log_dir, model_type, 'hparams'))

    original_state = copy.deepcopy(model.state_dict())

    results = []

    for i, combo in enumerate(combos):
        trial_cfg = copy.copy(base_cfg)
        trial_cfg.update(dict(zip(keys, combo)))

        exp_name  = _make_exp_name(model_type, keys, combo)
        combo_str = ' | '.join(f"{k}={v}" for k, v in zip(keys, combo))
        print(f"[{i+1:3d}/{total}] {combo_str}")
        print(f"    exp: {exp_name}")

        model.load_state_dict(original_state)

        t0 = time.time()
        best_r1 = _run_trial(
            model, train_img, train_txt, train_names, unique_names,
            val_loader, trial_cfg, device, model_type, num_epochs,
            writer=curves_writer, exp_name=exp_name,
        )
        elapsed = time.time() - t0

        # Log hparams + metric summary để so sánh trong tab HPARAMS
        hparam_dict = {k: float(v) if isinstance(v, (int, float)) else str(v)
                       for k, v in zip(keys, combo)}
        hparam_writer.add_hparams(
            hparam_dict,
            {'hparam/R@1': best_r1},
            run_name=exp_name,
        )

        row = dict(zip(keys, combo))
        row['val_R@1']  = best_r1
        row['time_sec'] = round(elapsed, 1)
        row['exp_name'] = exp_name
        results.append(row)

        print(f"      => Best R@1: {best_r1:.1f}%  ({elapsed:.0f}s)\n")

    curves_writer.close()
    hparam_writer.close()

    model.load_state_dict(original_state)

    results_df = pd.DataFrame(results).sort_values('val_R@1', ascending=False)
    csv_path   = os.path.join(save_dir, f'grid_search_{model_type}.csv')
    results_df.to_csv(csv_path, index=False)

    best_row = results_df.iloc[0]
    best_cfg = copy.copy(base_cfg)
    best_cfg.update({k: best_row[k] for k in keys})

    print(f"\n{'='*55}")
    print(f"  KẾT QUẢ GRID SEARCH — {model_type.upper()}")
    print(f"{'='*55}")
    print(results_df.drop(columns=['exp_name']).to_string(index=False))
    print(f"\n Best config:")
    for k in keys:
        print(f"    {k:20s} = {best_row[k]}")
    print(f"    {'val_R@1':20s} = {best_row['val_R@1']:.1f}%")
    print(f" CSV: {csv_path}")
    print(f"{'='*55}\n")

    return best_cfg, results_df