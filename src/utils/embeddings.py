import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# -------------------- EXTRACT FEATURES --------------------
def _extract_image_features(model, images):
    return model.image_encoder.extract_features(images)


def _extract_text_features(model, captions, lengths, model_type, attention_mask=None):
    text_enc = model.text_encoder

    if model_type in ('lstm', 'gru'):
        lengths = lengths.cpu().clamp(min=1)
        embeds  = text_enc.embedding_dropout(text_enc.embedding(captions))
        packed  = nn.utils.rnn.pack_padded_sequence(
            embeds, lengths, batch_first=True, enforce_sorted=False
        )
        outputs, _ = (text_enc.gru if model_type == 'gru' else text_enc.lstm)(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return text_enc.attn_pool(outputs, lengths)

    elif model_type == 'minilm':
        return text_enc.extract_features(captions, attention_mask)

    else:
        raise ValueError(f"model_type không hợp lệ: '{model_type}'! "
                         f"Chọn một trong: 'lstm', 'gru', 'minilm'")


# -------------------- UNPACK BATCH --------------------
def _unpack_batch(batch, device, model_type):
    if model_type in ('lstm', 'gru'):
        images, captions, lengths, names = batch
        attention_mask = None
    else:
        images, captions, lengths, names, attention_mask = batch
        attention_mask = attention_mask.to(device, non_blocking=True)

    return (
        images.to(device, non_blocking=True),
        captions.to(device, non_blocking=True),
        lengths, list(names), attention_mask,
    )


# -------------------- PRE-COMPUTE --------------------
def precompute_embeddings(model, loader, device, model_type):
    """Dùng cho val/test loader (eval_all_caps=True)"""
    model.eval()
    all_img, all_txt, all_names = [], [], []

    with torch.no_grad():
        for batch in loader:
            images, captions, lengths, names, attn_mask = _unpack_batch(batch, device, model_type)

            img_feat = model.image_encoder(images)

            if model_type in ('gru', 'lstm'):
                text_feat = model.text_encoder(captions, lengths)
            elif model_type == 'minilm':
                text_feat = model.text_encoder(captions, attn_mask)
            else:
                raise ValueError(f'Unsupported model_type: {model_type}')

            all_img.append(img_feat.cpu())
            all_txt.append(text_feat.cpu())
            all_names.extend(names)

    return torch.cat(all_img), torch.cat(all_txt), all_names


def precompute_train_embeddings(model, loader, device, model_type):
    """
    Dùng cho train loader (5 captions/ảnh).
    Trả về raw features chưa qua projection — projection được apply trong training loop.
    """
    model.eval()
    all_txt, all_names = [], []
    seen_imgs          = {}
    unique_names       = []

    with torch.no_grad():
        for batch in loader:
            images, captions, lengths, names, attn_mask = _unpack_batch(batch, device, model_type)

            img_feats = _extract_image_features(model, images).cpu()
            txt_feats = _extract_text_features(model, captions, lengths, model_type, attn_mask).cpu()

            all_txt.append(txt_feats)
            all_names.extend(names)

            for i, name in enumerate(names):
                if name not in seen_imgs:
                    seen_imgs[name] = img_feats[i]
                    unique_names.append(name)

    img_feats = torch.stack([seen_imgs[n] for n in unique_names])
    txt_feats = torch.cat(all_txt)

    print(f"  unique images : {img_feats.shape}")
    print(f"  all captions  : {txt_feats.shape}")

    return img_feats, txt_feats, all_names, unique_names


# -------------------- CACHED LOADER --------------------
def make_cached_loader(img_feats, txt_feats, names, unique_names, batch_size=128):
    name_to_idx = {n: i for i, n in enumerate(unique_names)}
    img_indices = torch.tensor([name_to_idx[n] for n in names], dtype=torch.long)
    txt_idx     = torch.arange(len(txt_feats))
    dataset     = TensorDataset(txt_idx, img_indices)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      pin_memory=True, num_workers=2)