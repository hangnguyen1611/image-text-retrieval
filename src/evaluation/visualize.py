import os
import textwrap
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from ..utils.embeddings import precompute_embeddings


def plot_history(history, title='Training History'):
    plt.figure(figsize=(8, 4))

    plt.plot(history['train_loss'], marker='o', label='Train Loss')
    plt.plot(history['val_loss'], marker='o', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def visualize_random_test_retrieval(model, test_loader, test_df, image_dir, device, model_type, n_top=10):
    """
    Lấy features từ precompute_embeddings (đã normalized 512-dim),
    random 1 query caption rồi hiển thị top-n kết quả retrieval.
    """

    # ------------------- Pre-compute toàn bộ features -------------------
    img_feats, txt_feats, image_names = precompute_embeddings(
        model, test_loader, device, model_type
    )

    # ------------------- Unique gallery -------------------
    seen, unique_names, unique_idx = {}, [], []
    for i, name in enumerate(image_names):
        if name not in seen:
            seen[name] = True
            unique_names.append(name)
            unique_idx.append(i)

    gallery    = img_feats[unique_idx]                        # [N_img, 512]
    name_to_gi = {name: i for i, name in enumerate(unique_names)}

    # ------------------- Random 1 query -------------------
    q_idx       = np.random.randint(len(image_names))
    q_feat      = txt_feats[q_idx]                            # [512]
    gt_name     = image_names[q_idx]
    q_caption   = test_df['caption'].iloc[q_idx]

    # ------------------- Similarity -------------------
    sims       = (gallery @ q_feat).cpu().numpy()             # [N_img]
    sorted_idx = np.argsort(sims)[::-1]
    top_idx    = sorted_idx[:n_top]

    gt_gi      = name_to_gi[gt_name]
    gt_rank    = int(np.where(sorted_idx == gt_gi)[0][0]) + 1
    gt_score   = float(sims[gt_gi])

    # ------------------- Caption map -------------------
    cap_map = test_df.groupby('image')['caption'].apply(list).to_dict()

    # ------------------- Plot -------------------
    cols    = min(5, n_top)
    rows    = (n_top + cols - 1) // cols
    fig     = plt.figure(figsize=(5 * (cols + 1), 5 * rows))
    gs      = gridspec.GridSpec(1, 2, width_ratios=[1, cols], wspace=0.05)

    # Ground truth
    ax_gt = fig.add_subplot(gs[0])
    try:
        ax_gt.imshow(Image.open(os.path.join(image_dir, gt_name)))
    except FileNotFoundError:
        ax_gt.text(0.5, 0.5, 'Missing', ha='center', va='center')
    ax_gt.set_title(
        f'GROUND TRUTH\nRank: {gt_rank}  Score: {gt_score:.3f}',
        fontsize=11, fontweight='bold', color='green'
    )
    for sp in ax_gt.spines.values():
        sp.set_color('green'); sp.set_linewidth(3)
    ax_gt.set_xticks([]); ax_gt.set_yticks([])

    # Top-N results
    gs_sub = gridspec.GridSpecFromSubplotSpec(rows, cols, subplot_spec=gs[1], hspace=0.4, wspace=0.1)
    for rank, gi in enumerate(top_idx):
        ax   = fig.add_subplot(gs_sub[rank])
        name = unique_names[gi]
        try:
            ax.imshow(Image.open(os.path.join(image_dir, name)))
        except FileNotFoundError:
            ax.text(0.5, 0.5, 'Missing', ha='center', va='center')

        is_correct   = (name == gt_name)
        border_color = 'green' if is_correct else 'red'
        for sp in ax.spines.values():
            sp.set_color(border_color)
            sp.set_linewidth(3 if is_correct else 1)

        cap = cap_map.get(name, [''])[0]
        ax.set_title(
            f'Top {rank+1}  ({sims[gi]:.3f})\n' +
            '\n'.join(textwrap.wrap(cap, width=30)),
            fontsize=7, color='green' if is_correct else 'black'
        )
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f'Query: "{q_caption.strip()}"', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.show()


def _encode_single_caption(model, caption, model_type, device, vocab=None, tokenizer=None, max_seq_len=64):
    """Encode 1 caption tự do -> normalized feature [1, 512]"""
    model.eval()
    with torch.no_grad():
        if model_type in ('lstm', 'gru'):
            ids    = vocab.numericalize(caption)
            cap_t  = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)  # [1, L]
            length = torch.tensor([len(ids)], dtype=torch.long)
            feat   = model.text_encoder(cap_t, length)

        elif model_type == 'minilm':
            enc   = tokenizer(
                caption, return_tensors='pt',
                padding=True, truncation=True, max_length=max_seq_len,
            )
            input_ids      = enc['input_ids'].to(device)
            attention_mask = enc['attention_mask'].to(device)
            feat = model.text_encoder(input_ids, attention_mask)

        else:
            raise ValueError(f'model_type không hợp lệ: {model_type}')

    return feat.cpu()  # [1, 512]


def make_retriever(model, test_loader, image_dir, device, model_type, vocab=None, tokenizer=None, max_seq_len=64):
    # ------------------- Pre-compute gallery 1 lần -------------------
    print('Pre-computing gallery...')
    img_feats, _, names = precompute_embeddings(model, test_loader, device, model_type)
    seen, unique_names, unique_idx = {}, [], []
    for i, n in enumerate(names):
        if n not in seen:
            seen[n] = True; unique_names.append(n); unique_idx.append(i)
    gallery_feats = img_feats[unique_idx]  # [N, 512]
    print(f'Gallery ready: {len(unique_names)} ảnh\n')

    def retrieve(query_caption, top_k=10):
        # ------------------- Encode query -------------------
        q_feat = _encode_single_caption(
            model, query_caption, model_type, device,
            vocab=vocab, tokenizer=tokenizer, max_seq_len=max_seq_len,
        )

        # ------------------- Similarity -------------------
        sims    = (gallery_feats @ q_feat.T).squeeze(1).numpy()
        top_idx = np.argsort(sims)[::-1][:top_k]

        # ------------------- Plot -------------------
        cols = min(5, top_k)
        rows = (top_k + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = np.array(axes).flatten()

        for rank, gi in enumerate(top_idx):
            ax   = axes[rank]
            name = unique_names[gi]
            try:
                ax.imshow(Image.open(os.path.join(image_dir, name)))
            except FileNotFoundError:
                ax.text(0.5, 0.5, 'Missing', ha='center', va='center',
                        transform=ax.transAxes)
            ax.set_title(f'Top {rank+1}  ({sims[gi]:.3f})', fontsize=9)
            for sp in ax.spines.values():
                sp.set_color('#2196F3')
                sp.set_linewidth(2)
            ax.set_xticks([])
            ax.set_yticks([])

        for i in range(top_k, len(axes)):
            axes[i].set_visible(False)

        fig.suptitle(f'Query: "{query_caption.strip()}"', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    return retrieve