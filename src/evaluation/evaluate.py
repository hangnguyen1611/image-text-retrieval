import torch
import os 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd 
from ..utils import precompute_embeddings

def evaluate(model, test_loader, cfg, device, model_type,
             model_name='cnn_lstm', output_dir='./output'):
    """
    Hàm đánh giá Text-to-Image Retrieval hiệu năng cao, xuất báo cáo và file phân tích lỗi.
    """
    model.eval()
    print(f'--> [Evaluation] Model: {model_name.upper()}')

    #-----------------PRECOMPUTE-----------------#
    img_feats, text_feats, image_names = precompute_embeddings(model, test_loader, device, model_type)
    img_feats  = img_feats.to(device)   
    text_feats = text_feats.to(device)

    #-----------------UNIQUE IMAGE-----------------#
    unique_names = []
    unique_img_feats = []

    seen = set()
    for feat, name in zip(img_feats, image_names):
        if name not in seen:
            seen.add(name)
            unique_names.append(name)
            unique_img_feats.append(feat)

    gallery = torch.stack(unique_img_feats).to(device)
    name_to_idx = {
        name: idx for idx, name in enumerate(unique_names)
    }

    #-----------------SIMILARITY-----------------#
    sim_matrix = text_feats @ gallery.T

    # Lấy top 10
    topk_k = 10
    topk_val, topk_idx = sim_matrix.topk(topk_k, dim=-1)
    topk_idx    = topk_idx.cpu().numpy()
    topk_scores = topk_val.cpu().numpy()

    r1, r5, r10, mrr = 0, 0, 0, 0.0
    error_cases = []

    for i, gt_name in enumerate(image_names):
        gt_id  = name_to_idx[gt_name]
        retrieved = topk_idx[i]

        rank = -1
        if gt_id in retrieved:
            rank = int(np.where(retrieved == gt_id)[0][0]) + 1
            if rank <= 1:  r1  += 1
            if rank <= 5:  r5  += 1
            if rank <= 10: r10 += 1
            mrr += 1.0 / rank

        # Lưu lại các trường hợp đoán lệch top 1 để phục vụ phân tích lỗi định tính (Qualitative Analysis)
        if rank != 1:
            pred_imgs = [
                unique_names[idx] for idx in retrieved[:3]
            ]
            pred_scores = [
                round(float(s), 4) for s in topk_scores[i][:3]
            ]
            error_cases.append({
                'ground_truth_image': gt_name,
                'actual_rank_of_gt': rank,
                'predicted_top3_images': pred_imgs,
                'predicted_top3_scores': pred_scores
            })

    total = len(image_names)

    # Tính toán tỷ lệ phần trăm Recall chuẩn hóa
    recall_1  = r1  / total * 100
    recall_5  = r5  / total * 100
    recall_10 = r10 / total * 100
    mean_mrr  = mrr / total

    report = f"""
==================================================
  BÁO CÁO ĐÁNH GIÁ - MODEL: {model_name.upper()}
==================================================
  Gallery (Ảnh độc lập) : {len(unique_names)} ảnh
  Queries (Câu truy vấn): {total} câu
  Recall@1             : {recall_1:.2f}%
  Recall@5             : {recall_5:.2f}%
  Recall@10            : {recall_10:.2f}%
  Mean MRR             : {mean_mrr:.4f}
==================================================
"""
    print(report)

    # Ghi báo cáo và file phân tích lỗi ra thư mục đầu ra
    output_dir = cfg.get('RESULT_DIR', './output')
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f'{model_name}_report.txt'), 'w') as f:
        f.write(report)

    if error_cases:
        pd.DataFrame(error_cases).to_csv(
            os.path.join(output_dir, f'test_{model_name}_error_analysis.csv'), index=False
        )

    return recall_1, recall_5, recall_10, mean_mrr