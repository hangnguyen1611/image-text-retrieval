import torch
from ..utils.embeddings import precompute_embeddings
from .clip_zeroshot import evaluate_clip_zeroshot
from .error_analysis import run_error_analysis
from .evaluate import evaluate
from .visualize import plot_history, visualize_random_test_retrieval, make_retriever

def _compute_similarity(model, img_feats, txt_feats, device):
    """Similarity matrix [n_txt, n_img] — features đã normalized từ precompute_embeddings"""
    img_emb = img_feats.to(device)
    txt_emb = txt_feats.to(device)
    return txt_emb @ img_emb.T


def _get_unique_images(names, img_feats):
    """Lấy unique image names + features theo thứ tự xuất hiện đầu tiên"""
    seen         = {}
    unique_names = []
    unique_idx   = []
    for i, name in enumerate(names):
        if name not in seen:
            seen[name] = True
            unique_names.append(name)
            unique_idx.append(i)
    return unique_names, img_feats[unique_idx]  


def _build_gt_matrix(cap_names, img_names):
    """
    gt[i, j] = True nếu caption i thuộc image j
    cap_names: [n_txt], img_names: [n_img]
    """
    img_to_idx = {name: idx for idx, name in enumerate(img_names)}
    gt = torch.zeros(len(cap_names), len(img_names), dtype=torch.bool)
    for i, name in enumerate(cap_names):
        j = img_to_idx.get(name)
        if j is not None:
            gt[i, j] = True
    return gt


def _recall_at_k(sim, gt, ks=(1, 5, 10)):
    """
    sim: [n_txt, n_img]
    gt:  [n_txt, n_img] bool
    """
    results = {}
    for k in ks:
        topk_idx = sim.topk(k, dim=1).indices         
        hit      = gt.gather(1, topk_idx).any(dim=1).float()
        results[f"R@{k}"] = round(hit.mean().item() * 100, 1)
    return results


def eval_standard(model, loader, device, model_type, ks=(1, 5, 10)): 
    """Eval đầy đủ — dùng cho cả frozen và unfrozen phase"""
    img_feats, txt_feats, names = precompute_embeddings(model, loader, device, model_type=model_type)

    unique_img_names, unique_img_feats = _get_unique_images(names, img_feats)

    sim = _compute_similarity(model, unique_img_feats, txt_feats, device)
    gt  = _build_gt_matrix(names, unique_img_names).to(sim.device)

    return _recall_at_k(sim, gt, ks)


def eval_cached(model, cached, split, device, ks=(1, 5, 10)):
    """Eval nhanh dùng pre-computed features."""
    img_feats = cached[split]["image"]
    txt_feats = cached[split]["text"]
    names     = cached[split]["names"]

    unique_img_names, unique_img_feats = _get_unique_images(names, img_feats)

    sim = _compute_similarity(model, unique_img_feats, txt_feats, device)
    gt  = _build_gt_matrix(names, unique_img_names).to(sim.device)

    return _recall_at_k(sim, gt, ks)