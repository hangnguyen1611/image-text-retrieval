import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel


# ------------------- Feature extraction -------------------
@torch.no_grad()
def _encode_images(model, processor, image_paths, device, batch_size=64):
    """Encode toàn bộ ảnh unique -> normalized features [N, D]."""
    all_feats = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc='Encoding images'):
        batch_paths = image_paths[i: i + batch_size]
        images = [Image.open(p).convert('RGB') for p in batch_paths]
        inputs = processor(images=images, return_tensors='pt', padding=True).to(device)
        feats  = model.get_image_features(**inputs)
        if hasattr(feats, 'pooler_output'):
            feats = feats.pooler_output
        feats  = F.normalize(feats, p=2, dim=1)
        all_feats.append(feats.cpu())
    return torch.cat(all_feats)


@torch.no_grad()
def _encode_texts(model, processor, captions, device, batch_size=256):
    """Encode toàn bộ captions -> normalized features [N, D]"""
    all_feats = []
    for i in tqdm(range(0, len(captions), batch_size), desc='Encoding texts'):
        batch_caps = captions[i: i + batch_size]
        inputs = processor(
            text=batch_caps, return_tensors='pt',
            padding=True, truncation=True, max_length=77,
        ).to(device)
        feats = model.get_text_features(**inputs)
        if hasattr(feats, 'pooler_output'):
            feats = feats.pooler_output
        feats = F.normalize(feats, p=2, dim=1)
        all_feats.append(feats.cpu())
    return torch.cat(all_feats)


# ------------------- Metrics -------------------
def _compute_metrics(sim_matrix, image_names, unique_names):
    """
    sim_matrix : [n_queries, n_gallery] tensor
    image_names: [n_queries] tên ảnh gt của từng query
    unique_names: [n_gallery] tên ảnh trong gallery

    Trả về dict R@1, R@5, R@10, MRR và danh sách error_cases.
    """
    name_to_idx = {name: idx for idx, name in enumerate(unique_names)}
    topk_val, topk_idx = sim_matrix.topk(10, dim=1)
    topk_idx    = topk_idx.cpu().numpy()
    topk_scores = topk_val.cpu().numpy()

    r1 = r5 = r10 = 0
    mrr = 0.0
    error_cases = []

    for i, gt_name in enumerate(image_names):
        gt_id     = name_to_idx[gt_name]
        retrieved = topk_idx[i]

        rank = -1
        if gt_id in retrieved:
            rank = int(np.where(retrieved == gt_id)[0][0]) + 1
            if rank <= 1:  r1  += 1
            if rank <= 5:  r5  += 1
            if rank <= 10: r10 += 1
            mrr += 1.0 / rank

        if rank != 1:
            error_cases.append({
                'ground_truth_image'  : gt_name,
                'actual_rank_of_gt'   : rank,
                'predicted_top3_images': [unique_names[idx] for idx in retrieved[:3]],
                'predicted_top3_scores': [round(float(s), 4) for s in topk_scores[i][:3]],
            })

    total = len(image_names)
    return {
        'R@1' : round(r1  / total * 100, 2),
        'R@5' : round(r5  / total * 100, 2),
        'R@10': round(r10 / total * 100, 2),
        'MRR' : round(mrr / total, 4),
    }, error_cases


# ------------------- Main -------------------
def evaluate_clip_zeroshot(image_dir, test_df, device, clip_model='openai/clip-vit-base-patch32', output_dir='./output', batch_size=64):
    model_tag = clip_model.split('/')[-1]
    print(f'\n--> [Zero-Shot] Model: {model_tag.upper()}')

    # ------------------- Load CLIP -------------------
    print('Loading CLIP model...')
    clip  = CLIPModel.from_pretrained(clip_model).to(device).eval()
    proc  = CLIPProcessor.from_pretrained(clip_model)

    # ------------------- Chuẩn bị dữ liệu -------------------
    # Gallery: unique images
    unique_names = list(dict.fromkeys(test_df['image'].tolist()))
    image_paths  = [os.path.join(image_dir, n) for n in unique_names]

    # Queries: tất cả captions
    captions    = test_df['caption'].tolist()
    image_names = test_df['image'].tolist()

    # ------------------- Encode -------------------
    img_feats  = _encode_images(clip, proc, image_paths, device, batch_size).to(device)
    text_feats = _encode_texts(clip, proc, captions,    device, batch_size).to(device)

    # ------------------- Similarity & metrics -------------------
    sim_matrix = text_feats @ img_feats.T          # [n_queries, n_gallery]
    metrics, error_cases = _compute_metrics(sim_matrix, image_names, unique_names)

    # ------------------- Report -------------------
    report = f"""
    ==================================================
    BÁO CÁO ĐÁNH GIÁ - MODEL: {model_tag.upper()} (ZERO-SHOT)
    ==================================================
    Gallery (Ảnh độc lập) : {len(unique_names)} ảnh
    Queries (Câu truy vấn): {len(captions)} câu
    Recall@1             : {metrics['R@1']:.2f}%
    Recall@5             : {metrics['R@5']:.2f}%
    Recall@10            : {metrics['R@10']:.2f}%
    Mean MRR             : {metrics['MRR']:.4f}
    ==================================================
    """
    print(report)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f'{model_tag}_zeroshot_report.txt'), 'w') as f:
        f.write(report)

    if error_cases:
        pd.DataFrame(error_cases).to_csv(
            os.path.join(output_dir, f'test_{model_tag}_zeroshot_error_analysis.csv'),
            index=False,
        )

    return metrics