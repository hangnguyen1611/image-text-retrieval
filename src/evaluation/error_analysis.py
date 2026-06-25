import os
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from ..utils.embeddings import precompute_embeddings


# -------------- Helpers --------------
def _build_gallery(img_feats, image_names):
    """Lấy unique image features theo thứ tự xuất hiện đầu tiên"""
    seen, unique_names, unique_idx = {}, [], []
    for i, name in enumerate(image_names):
        if name not in seen:
            seen[name] = True
            unique_names.append(name)
            unique_idx.append(i)
    gallery = img_feats[unique_idx]            # [N_img, D]
    name_to_gi = {n: i for i, n in enumerate(unique_names)}
    return gallery, unique_names, name_to_gi


def _compute_error_records(txt_feats, gallery, image_names, unique_names, name_to_gi, top_k=10):
    """
    Tính similarity matrix, lấy top-K, trả về danh sách record cho mọi query để phục vụ phân tích đầy đủ.

    Mỗi record:
        caption_idx, gt_name, rank (1-based; -1 nếu GT nằm ngoài top_k),
        gt_score (cosine với GT image), top1_score,
        top_k_names, top_k_scores
    """
    sim_matrix = txt_feats @ gallery.T                      # [N_txt, N_img]
    topk_val, topk_idx = sim_matrix.topk(top_k, dim=1)
    topk_idx    = topk_idx.cpu().numpy()
    topk_scores = topk_val.cpu().numpy()

    # GT score trực tiếp từ sim_matrix 
    gt_gi_list = [name_to_gi[n] for n in image_names]
    gt_scores  = sim_matrix[range(len(image_names)), gt_gi_list].cpu().numpy()

    records = []
    for i, gt_name in enumerate(image_names):
        gt_gi     = name_to_gi[gt_name]
        retrieved = topk_idx[i]

        rank = -1
        if gt_gi in retrieved:
            rank = int(np.where(retrieved == gt_gi)[0][0]) + 1

        records.append({
            'caption_idx'   : i,
            'gt_name'       : gt_name,
            'rank'          : rank,
            'gt_score'      : float(gt_scores[i]),
            'top1_score'    : float(topk_scores[i][0]),
            'top_k_names'   : [unique_names[idx] for idx in retrieved],
            'top_k_scores'  : [float(s) for s in topk_scores[i]],
        })
    return records


# -------------- Phân tích định lượng --------------
def _quantitative_analysis(records, model_name, output_dir):
    """
    Thống kê và vẽ biểu đồ:
      - Rank distribution (histogram)
      - Tỷ lệ hit@1/5/10, hard failure (rank == -1)
      - GT score vs top-1 score (box plot / scatter)
      - Score gap = top1_score - gt_score (phân phối)
    Xuất: <model_name>_quantitative.png và dict chỉ số tóm tắt.
    """
    ranks      = np.array([r['rank']      for r in records])
    gt_scores  = np.array([r['gt_score']  for r in records])
    top1_scores= np.array([r['top1_score']for r in records])
    score_gaps = top1_scores - gt_scores    # > 0 nghĩa là top-1 beat GT

    total       = len(records)
    hit1        = int((ranks == 1).sum())
    hit5        = int(((ranks >= 1) & (ranks <= 5)).sum())
    hit10       = int(((ranks >= 1) & (ranks <= 10)).sum())
    hard_fail   = int((ranks == -1).sum())   # GT không có trong top-10
    error_count = total - hit1               # tất cả query không hit@1

    summary = {
        'total_queries'  : total,
        'R@1 (%)'        : round(hit1  / total * 100, 2),
        'R@5 (%)'        : round(hit5  / total * 100, 2),
        'R@10 (%)'       : round(hit10 / total * 100, 2),
        'error_count'    : error_count,
        'hard_failure'   : hard_fail,
        'hard_failure (%)': round(hard_fail / total * 100, 2),
        'mean_gt_score'  : round(float(gt_scores.mean()),  4),
        'mean_top1_score': round(float(top1_scores.mean()),4),
        'mean_score_gap' : round(float(score_gaps.mean()), 4),
    }

    # -------------- In báo cáo text --------------
    sep = '=' * 52
    print(f'\n{sep}')
    print(f'  PHÂN TÍCH ĐỊNH LƯỢNG LỖI — {model_name.upper()}')
    print(sep)
    for k, v in summary.items():
        print(f'  {k:<22}: {v}')
    print(sep)

    # -------------- Vẽ figure 2×2 --------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f'Định lượng lỗi — {model_name.upper()}',
                 fontsize=14, fontweight='bold')

    # Rank distribution histogram (chỉ các query có rank != -1)
    ax = axes[0, 0]
    valid_ranks = ranks[ranks != -1]
    ax.hist(valid_ranks, bins=range(1, 13), color='steelblue',
            edgecolor='white', linewidth=0.6, align='left')
    ax.axvline(x=1,  color='green',  linestyle='--', linewidth=1.2, label='Rank 1')
    ax.axvline(x=5,  color='orange', linestyle='--', linewidth=1.2, label='Rank 5')
    ax.axvline(x=10, color='red',    linestyle='--', linewidth=1.2, label='Rank 10')
    ax.set_xlabel('Rank của GT image')
    ax.set_ylabel('Số lượng query')
    ax.set_title('Rank Distribution (rank 1–10)')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # Pie chart: hit@1 / hit@2-5 / hit@6-10 / hard fail
    ax = axes[0, 1]
    hit_1_only  = hit1
    hit_2_5     = int(((ranks >= 2) & (ranks <= 5)).sum())
    hit_6_10    = int(((ranks >= 6) & (ranks <= 10)).sum())
    sizes  = [hit_1_only, hit_2_5, hit_6_10, hard_fail]
    labels = [f'Rank 1\n({hit_1_only})',
              f'Rank 2–5\n({hit_2_5})',
              f'Rank 6–10\n({hit_6_10})',
              f'Hard Fail\n({hard_fail})']
    colors = ['#4CAF50', '#FFC107', '#FF7043', '#B0BEC5']
    wedges, _, autotexts = ax.pie(
        sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=140,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
    )
    for at in autotexts:
        at.set_fontsize(8)
    ax.set_title('Phân bổ kết quả retrieval')

    # Box plot: GT score vs Top-1 score
    ax = axes[1, 0]
    ax.boxplot(
        [gt_scores, top1_scores],
        labels=['GT Score', 'Top-1 Score'],
        patch_artist=True,
        boxprops=dict(facecolor='lightblue', color='navy'),
        medianprops=dict(color='red', linewidth=2),
        whiskerprops=dict(color='navy'),
        capprops=dict(color='navy'),
        flierprops=dict(marker='o', markersize=3, alpha=0.4, color='gray'),
    )
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('GT Score vs Top-1 Score')
    ax.grid(axis='y', alpha=0.3)

    # Histogram score gap (top1 - gt)
    ax = axes[1, 1]
    ax.hist(score_gaps, bins=40, color='tomato', edgecolor='white',
            linewidth=0.5, alpha=0.85)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.2, label='Gap = 0')
    ax.set_xlabel('Score Gap (Top-1 − GT)')
    ax.set_ylabel('Số lượng query')
    ax.set_title('Phân phối Score Gap')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{model_name}_quantitative.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'  [Saved] {save_path}')

    return summary


# -------------- Phân tích định tính --------------
def _load_img(path: str):
    """Load ảnh PIL, trả về None nếu không tìm thấy"""
    try:
        return Image.open(path).convert('RGB')
    except (FileNotFoundError, OSError):
        return None


def _qualitative_analysis(records, captions, image_dir, model_name, output_dir, n=12):
    """
    Chọn n error cases (không hit@1) -> vẽ grid:
      Cột 0  : GT image + caption query
      Cột 1–3: Top-3 predicted images (viền đỏ/xanh)

    Xuất: <model_name>_qualitative_errors.png
    """
    error_records = [r for r in records if r['rank'] != 1]
    if not error_records:
        print('  [Qualitative] Không có error case nào!')
        return

    # Chọn mẫu đại diện: ưu tiên hard fail trước, sau đó rank cao
    hard   = [r for r in error_records if r['rank'] == -1]
    others = sorted([r for r in error_records if r['rank'] != -1],
                    key=lambda x: x['rank'], reverse=True)
    chosen = (hard + others)[:n]

    n_cols  = 4    # GT + top3
    n_rows  = len(chosen)
    fig_h   = 3.2 * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, fig_h))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(f'Phân tích định tính lỗi — {model_name.upper()}',
                 fontsize=13, fontweight='bold')

    for row, rec in enumerate(chosen):
        caption   = captions[rec['caption_idx']] if rec['caption_idx'] < len(captions) else ''
        rank_str  = str(rec['rank']) if rec['rank'] != -1 else '>10'

        # -------------- Cột 0: GT --------------
        ax = axes[row, 0]
        img = _load_img(os.path.join(image_dir, rec['gt_name']))
        if img:
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, 'Missing', ha='center', va='center',
                    transform=ax.transAxes, fontsize=9)
            ax.set_facecolor('#f0f0f0')
        ax.set_title(
            f'GT  (Rank: {rank_str})\nScore: {rec["gt_score"]:.3f}',
            fontsize=8, fontweight='bold', color='#1B5E20'
        )
        for sp in ax.spines.values():
            sp.set_color('#43A047'); sp.set_linewidth(3)
        ax.set_xticks([]); ax.set_yticks([])

        # Caption bên dưới (wrapped)
        wrapped = '\n'.join(textwrap.wrap(caption, width=32))
        ax.set_xlabel(wrapped, fontsize=6.5, labelpad=4, color='#333')

        # -------------- Cột 1–3: Top-3 predicted --------------
        for col in range(1, 4):
            ax = axes[row, col]
            if col - 1 >= len(rec['top_k_names']):
                ax.set_visible(False)
                continue

            pred_name  = rec['top_k_names'][col - 1]
            pred_score = rec['top_k_scores'][col - 1]
            is_correct = (pred_name == rec['gt_name'])

            img = _load_img(os.path.join(image_dir, pred_name))
            if img:
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, 'Missing', ha='center', va='center',
                        transform=ax.transAxes, fontsize=9)
                ax.set_facecolor('#f9f9f9')

            border = '#43A047' if is_correct else '#E53935'
            lw     = 3 if is_correct else 1.5
            ax.set_title(
                f'Top-{col}  ({pred_score:.3f})',
                fontsize=8,
                color='#1B5E20' if is_correct else '#B71C1C'
            )
            for sp in ax.spines.values():
                sp.set_color(border); sp.set_linewidth(lw)
            ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_path = os.path.join(output_dir, f'{model_name}_qualitative_errors.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'  [Saved] {save_path}')


# -------------- Phân tích theo nhóm (score-based) --------------
def _group_analysis(records, model_name, output_dir):
    """
    Phân nhóm lỗi theo GT score của query:
      - Nhóm 1 "Rất thấp"  : gt_score < Q25
      - Nhóm 2 "Thấp"      : Q25 <= gt_score < Q50
      - Nhóm 3 "Trung bình": Q50 <= gt_score < Q75
      - Nhóm 4 "Cao"       : gt_score >= Q75

    Với mỗi nhóm thống kê: số lượng, tỷ lệ hit@1/5/10, hard failure.
    Xuất: <model_name>_group_analysis.png  +  <model_name>_group_stats.csv
    """
    gt_scores = np.array([r['gt_score'] for r in records])
    ranks     = np.array([r['rank']     for r in records])

    q25, q50, q75 = np.percentile(gt_scores, [25, 50, 75])
    boundaries = [(-np.inf, q25), (q25, q50), (q50, q75), (q75, np.inf)]
    group_names = [
        f'Rất thấp\n(< {q25:.3f})',
        f'Thấp\n([{q25:.3f}, {q50:.3f}))',
        f'Trung bình\n([{q50:.3f}, {q75:.3f}))',
        f'Cao\n(≥ {q75:.3f})',
    ]
    colors = ['#EF5350', '#FFA726', '#66BB6A', '#42A5F5']

    rows = []
    for (lo, hi), gname in zip(boundaries, group_names):
        mask   = (gt_scores >= lo) & (gt_scores < hi) if hi != np.inf \
                 else gt_scores >= lo
        idx    = np.where(mask)[0]
        if len(idx) == 0:
            continue
        g_ranks = ranks[idx]
        total_g = len(idx)
        h1  = int((g_ranks == 1).sum())
        h5  = int(((g_ranks >= 1) & (g_ranks <= 5)).sum())
        h10 = int(((g_ranks >= 1) & (g_ranks <= 10)).sum())
        hf  = int((g_ranks == -1).sum())
        rows.append({
            'group'         : gname.replace('\n', ' '),
            'count'         : total_g,
            'R@1 (%)'       : round(h1  / total_g * 100, 1),
            'R@5 (%)'       : round(h5  / total_g * 100, 1),
            'R@10 (%)'      : round(h10 / total_g * 100, 1),
            'hard_fail'     : hf,
            'hard_fail (%)' : round(hf / total_g * 100, 1),
            'mean_gt_score' : round(float(gt_scores[idx].mean()), 4),
        })

    df_group = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, f'{model_name}_group_stats.csv')
    df_group.to_csv(csv_path, index=False)

    print(f'\n  [Group Analysis] {model_name.upper()}')
    print(df_group.to_string(index=False))
    print(f'  [Saved] {csv_path}')

    # -------------- Grouped bar chart --------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Phân tích lỗi theo nhóm similarity — {model_name.upper()}',
                 fontsize=13, fontweight='bold')

    x      = np.arange(len(df_group))
    width  = 0.26
    labels = [r['group'] for r in rows]

    # Recall bars
    ax = axes[0]
    ax.bar(x - width, df_group['R@1 (%)'],  width, label='R@1',  color='#42A5F5')
    ax.bar(x,         df_group['R@5 (%)'],  width, label='R@5',  color='#66BB6A')
    ax.bar(x + width, df_group['R@10 (%)'], width, label='R@10', color='#FFA726')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Recall (%)')
    ax.set_title('Recall@K theo nhóm GT score')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    for bar in ax.patches:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                    f'{h:.0f}', ha='center', va='bottom', fontsize=7)

    # Hard failure bars
    ax = axes[1]
    bar_colors = [colors[i % len(colors)] for i in range(len(df_group))]
    bars = ax.bar(x, df_group['hard_fail (%)'], color=bar_colors, edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Hard Failure (%)')
    ax.set_title('Tỷ lệ Hard Failure (GT nằm ngoài top-10)')
    ax.grid(axis='y', alpha=0.3)
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                    f'{h:.1f}%', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{model_name}_group_analysis.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'  [Saved] {save_path}')

    return df_group


# -------------- Export error CSV đầy đủ --------------
def _export_error_csv(records: list[dict], captions: list,
                      model_name: str, output_dir: str):
    """Xuất CSV chi tiết tất cả error cases (rank != 1)."""
    rows = []
    for r in records:
        if r['rank'] == 1:
            continue
        rows.append({
            'caption_idx'           : r['caption_idx'],
            'caption'               : captions[r['caption_idx']] if r['caption_idx'] < len(captions) else '',
            'ground_truth_image'    : r['gt_name'],
            'actual_rank_of_gt'     : r['rank'],
            'gt_score'              : round(r['gt_score'],   4),
            'top1_score'            : round(r['top1_score'], 4),
            'score_gap'             : round(r['top1_score'] - r['gt_score'], 4),
            'predicted_top3_images' : str(r['top_k_names'][:3]),
            'predicted_top3_scores' : str([round(s, 4) for s in r['top_k_scores'][:3]]),
        })
    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, f'{model_name}_error_analysis.csv')
    df.to_csv(path, index=False)
    print(f'  [Saved] error CSV ({len(rows)} rows) → {path}')
    return df


# -------------- Entry point chính --------------
def run_error_analysis(model, test_loader, image_dir, device, model_type,
                       model_name='model', output_dir='./output', top_k=10, n_qualitative=12):
    os.makedirs(output_dir, exist_ok=True)

    # -------------- Precompute embeddings --------------
    print(f'\n[ErrorAnalysis] Precomputing embeddings — {model_name.upper()} ...')
    img_feats, txt_feats, image_names = precompute_embeddings(
        model, test_loader, device, model_type=model_type
    )

    # -------------- Captions đúng thứ tự với image_names --------------
    captions = [cap for _, cap in test_loader.dataset.pairs]

    # -------------- Build gallery --------------
    gallery, unique_names, name_to_gi = _build_gallery(img_feats, image_names)
    print(f'  Gallery: {len(unique_names)} images | Queries: {len(image_names)}')

    # -------------- Compute error records --------------
    records = _compute_error_records(
        txt_feats, gallery, image_names, unique_names, name_to_gi, top_k=top_k
    )

    # -------------- Định lượng --------------
    print('\n[1/3] Phân tích định lượng...')
    summary = _quantitative_analysis(records, model_name, output_dir)

    # -------------- Định tính --------------
    print(f'\n[2/3] Phân tích định tính ({n_qualitative} cases)...')
    _qualitative_analysis(records, captions, image_dir,
                          model_name, output_dir, n=n_qualitative)

    # -------------- Theo nhóm --------------
    print('\n[3/3] Phân tích theo nhóm similarity score...')
    group_stats = _group_analysis(records, model_name, output_dir)

    # -------------- Export CSV --------------
    error_df = _export_error_csv(records, captions, model_name, output_dir)

    print(f'\n[ErrorAnalysis] Hoàn tất! Kết quả lưu tại: {output_dir}')
    return {
        'summary'    : summary,
        'group_stats': group_stats,
        'error_df'   : error_df,
    }