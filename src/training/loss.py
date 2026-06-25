import numpy as np
import torch
import torch.nn.functional as F

def contrastive_loss(img_emb, txt_emb, image_names, temperature=0.1, margin=0.2, hard_neg_weight=0.3):
    device = img_emb.device
    scale  = 1.0 / temperature
    sim    = img_emb @ txt_emb.T * scale  # scaled sim, dùng cho InfoNCE

    _, image_ids = np.unique(image_names, return_inverse=True)
    ids          = torch.tensor(image_ids, device=device, dtype=torch.long)
    pos_mask     = (ids.unsqueeze(0) == ids.unsqueeze(1))  # [B, B]

    # Soft target InfoNCE (dùng scaled sim)
    targets_i2t = pos_mask.float()
    targets_i2t = targets_i2t / targets_i2t.sum(dim=1, keepdim=True).clamp(min=1e-8)

    targets_t2i = pos_mask.T.float()
    targets_t2i = targets_t2i / targets_t2i.sum(dim=1, keepdim=True).clamp(min=1e-8)

    loss_i2t = -(F.log_softmax(sim,   dim=1) * targets_i2t).sum(dim=1).mean()
    loss_t2i = -(F.log_softmax(sim.T, dim=1) * targets_t2i).sum(dim=1).mean()
    infonce_loss = (loss_i2t + loss_t2i) / 2

    # Hard negative: tính trên cosine space (chưa scale)
    cos_sim = img_emb @ txt_emb.T   # cosine similarity, range [-1, 1]
    min_val = torch.finfo(cos_sim.dtype).min

    pos_sim_i2t  = (cos_sim   * pos_mask.float()).sum(1) / pos_mask.float().sum(1)
    pos_sim_t2i  = (cos_sim.T * pos_mask.T.float()).sum(1) / pos_mask.T.float().sum(1)

    hard_neg_i2t = cos_sim.masked_fill(pos_mask,     min_val).max(dim=1)[0]
    hard_neg_t2i = cos_sim.T.masked_fill(pos_mask.T, min_val).max(dim=1)[0]

    hard_neg_loss = (
        F.relu(margin + hard_neg_i2t - pos_sim_i2t).mean() +
        F.relu(margin + hard_neg_t2i - pos_sim_t2i).mean()
    ) / 2
    return infonce_loss + hard_neg_weight * hard_neg_loss