# Flickr8k Image-Text Retrieval (Dual-Encoder + Contrastive Learning)

Dự án xây dựng mô hình **image-text retrieval** trên tập **Flickr8k**, sử dụng kiến trúc **dual-encoder** (1 encoder ảnh + 1 encoder văn bản) huấn luyện bằng **contrastive learning** (symmetric InfoNCE + hard negative mining).

Hỗ trợ 3 loại text encoder, chọn qua `MODEL_TYPE`:

| `MODEL_TYPE` | Text encoder              | Mô tả |
|--------------|----------------------------|-------|
| `lstm`       | BiLSTM (2 layer) + attention pooling | Train từ đầu |
| `gru`        | BiGRU (2 layer) + attention pooling  | Train từ đầu |
| `minilm`     | `sentence-transformers/all-MiniLM-L6-v2` (frozen) + projection head | Pretrained, chỉ train projection |

Image encoder: **ResNet50** (pretrained ImageNet) + projection head, có thể freeze/fine-tune một phần backbone.

*Tải dữ liệu flickr8k tại đây: https://www.kaggle.com/datasets/adityajn105/flickr8k*

---

## 1. Cấu trúc thư mục

```
project/
├── configs/
│   └── config.py   
│        
├── data/
│   ├── dataset.py             # FlickrDataset, collate_fn 
│   ├── loaders.py             # Tạo train/val/test DataLoader + transform
│   ├── preprocessed/          # Chưa các file csv sau khi preprocess
│   ├── raw/
│   │   ├── Image.zip          # File zip chứa ảnh
│   │   └── caption.txt        # File txt chứa các caption tương ứng cho mỗi ảnh
│   └── vocab/
│       ├── tokenizer.py       # class Vocabulary 
│       └── vocab.json         # vocab đã build sẵn
│
├── outputs/                   
│   ├── checkpoints/           # Lưu checkpoints
│   └── results/               # Lưu các kết quả đánh giá và phân tích lỗi
│
├── notebooks/
│   ├── eda.ipynb              # EDA dữ liệu
│   ├── evaluate.ipynb         # Đánh giá mô hình
│   ├── grid_search.ipynb      # Tối ưu tham số
│   ├── train.ipynb            # Train main
│   └── prepare.ipynb          # Tiền xử lý, chuẩn bị vocab và split data
│
├── src/
│   ├── model/
│   │   ├── image_encoder.py   # ImageEncoder (ResNet50 + projection)
│   │   ├── text_encoders.py   # TextEncoderGRU / TextEncoderLSTM / TextEncoderMiniLM
│   │   └── contrastive.py     # ContrastiveModel: gộp image + text encoder
│   ├── training/
│   │   ├── loss.py           
│   │   ├── train.py           
│   │   └── grid_search.py     
│   ├── evaluation/
│   │   ├── __init__.py        # Các hàm hỗ trợ eval
│   │   ├── evaluate.py        
│   │   ├── clip_zeroshot.py   # CLIP zero-shot baseline
│   │   ├── error_analysis.py    
│   │   └── visualize.py        
│   ├── preprocessing/
│   │   └── preprocess.py        
│   └── utils/
│       └── embeddings.py      # precompute_embeddings / precompute_train_embeddings, cached loader
│
├── requirements.txt
└── README.md
```

---

## 2. Cấu hình (`configs/config.py`)

Mỗi model type (`cnn_lstm`, `cnn_gru`, `cnn_minilm`) có 1 dict config riêng, lấy qua:

```python
from configs.config import get_config, get_search_space

cfg = get_config('cnn_gru')          # hoặc 'cnn_lstm' / 'cnn_minilm'
search_space = get_search_space('cnn_gru')
```

Các tham số chính trong config:

| Key | Ý nghĩa |
|-----|---------|
| `MODEL_TYPE` | `'lstm'`/ `'gru'`/`'minilm'` |
| `BATCH_SIZE`, `MAX_LEN` | batch size, độ dài caption tối đa |
| `LEARNING_RATE`, `FINETUNE_LR`, `WEIGHT_DECAY` | LR cho projection / backbone, weight decay (AdamW) |
| `WARMUP_EPOCHS`, `NUM_EPOCHS` | số epoch warmup LR (cosine schedule) và tổng epoch |
| `PROJ_WARMUP_EPOCHS` | số epoch train **chỉ projection** trước khi vào main loop (cần backbone frozen) |
| `EMBED_DIM`, `HIDDEN_DIM`, `DROPOUT` | kích thước embedding chung, hidden size GRU/LSTM, dropout |
| `TEMPERATURE`, `MARGIN`, `HARD_NEG_WEIGHT` | tham số contrastive loss (xem mục 4) |
| `FREEZE_BACKBONE` | freeze ResNet50 backbone lúc khởi tạo |
| `UNFREEZE_FROM` | layer ResNet bắt đầu unfreeze (`'layer3'` ⇒ mở `layer3` + `layer4`) |
| `UNFREEZE_EPOCH` | epoch sẽ unfreeze backbone (xem mục 5) |

> **Lưu ý**: `UNFREEZE_EPOCH` trong config **không tự động được áp dụng**.
> `train_model()` nhận `unfreeze_epoch` là argument riêng (mặc định `None`). 
>**Phải** truyền tay khi gọi training — *xem ví dụ ở mục 5*.

---

## 3. Pipeline sử dụng

### 3.1. Tiền xử lý dữ liệu

```python
from src.preprocessing.preprocess import DataSplitter, CaptionProcessor

processor = CaptionProcessor()
df_clean  = processor.process_dataframe(df_raw)        # lowercase + bỏ dấu câu

splitter  = DataSplitter(seed=42)
train_df, val_df, test_df = splitter.split_karpathy(df_clean)  # 6000/1000/1000 ảnh
splitter.save_splits(train_df, val_df, test_df, output_dir='data/processed')
```

### 3.2. Vocab (cho `lstm`/`gru`)

```python
from data.vocab.tokenizer import Vocabulary

vocab = Vocabulary(min_freq=5)
vocab.build_vocab(train_df)
vocab.save('data/vocab/vocab.json')
```

### 3.3. DataLoader

```python
from data.loaders import build_loaders
from configs.config import get_config

cfg = get_config('cnn_gru')
train_loader, val_loader, test_loader = build_loaders(
    train_df, val_df, test_df, vocab,
    image_dir='data/raw/Images',
    cfg=cfg,
    model_type=cfg['MODEL_TYPE'],
    tokenizer=None   # bắt buộc truyền tokenizer HF nếu model_type='minilm'
)
```

### 3.4. Khởi tạo model

```python
from src.model.image_encoder import ImageEncoder
from src.model.text_encoders import TextEncoderGRU
from src.model.contrastive import ContrastiveModel

image_encoder = ImageEncoder(cfg)
text_encoder  = TextEncoderGRU(vocab_size=len(vocab), cfg=cfg)
model = ContrastiveModel(image_encoder, text_encoder, cfg).to(cfg['DEVICE'])
```

### 3.5. Training

```python
from src.training.train import train_model
from configs.config import get_save_path, update_checkpoint_r1

save_path = get_save_path('cnn_gru', cfg_tag='exp1')

history = train_model(
    model, train_loader, val_loader, cfg, cfg['DEVICE'],
    model_type=cfg['MODEL_TYPE'],
    name_exp='cnn_gru_exp1',
    save_path=save_path,
    unfreeze_epoch=cfg['UNFREEZE_EPOCH'],   # Phải truyền tay 
    patience=5,
    eval_every=2,
    proj_warmup_epochs=cfg['PROJ_WARMUP_EPOCHS']
)
```

### 3.6. Đánh giá / phân tích lỗi

```python
from src.evaluation import evaluate, eval_standard, run_error_analysis, evaluate_clip_zeroshot

r1, r5, r10, mrr = evaluate(model, test_loader, cfg, cfg['DEVICE'],
                             model_type=cfg['MODEL_TYPE'], model_name='cnn_gru')

results = run_error_analysis(
    model, test_loader, test_df, image_dir='data/raw/Images',
    device=cfg['DEVICE'], model_type=cfg['MODEL_TYPE'],
    model_name='cnn_gru', output_dir='outputs/results'
)

# Baseline CLIP zero-shot
clip_metrics = evaluate_clip_zeroshot(
    image_dir='data/raw/Images', test_df=test_df, device=cfg['DEVICE']
)
```

### 3.7. Grid search

```python
from src.training.grid_search import grid_search
from configs.config import get_search_space

search_space = get_search_space('cnn_gru')
num_epochs = search_space.pop('NUM_EPOCHS')
param_grid = search_space

best_cfg, results_df = grid_search(
    model, train_loader, val_loader, cfg, cfg['DEVICE'],
    model_type=cfg['MODEL_TYPE'],
    num_epochs=num_epochs,
    param_grid=param_grid,
)
```

> Grid search chỉ train **projection heads** trên feature cache (backbone frozen tại thời điểm gọi). Nên chạy grid search **trước** khi unfreeze backbone, để feature cache phản ánh đúng backbone pretrained ban đầu.

---

## 4. Hàm loss (`src/training/loss.py`)

`contrastive_loss(img_emb, txt_emb, image_names, temperature, margin, hard_neg_weight)`:

- **InfoNCE (symmetric)**: soft-target trên các cặp positive cùng `image_name` (mỗi ảnh có 5 caption), scale similarity bằng `1/temperature`.
- **Hard negative mining**: tính trên cosine similarity (chưa scale), phạt theo `relu(margin + hard_neg_sim - pos_sim)` theo cả 2 chiều i2t/t2i.
- Tổng loss = `infonce_loss + hard_neg_weight * hard_neg_loss`.

---

## 5. Cơ chế freeze / unfreeze backbone (`train.py`)

- `FREEZE_BACKBONE=True` lúc khởi tạo `ImageEncoder` ⇒ backbone ResNet50 frozen, cho phép **precompute & cache** image/text features (tăng tốc training rất nhiều khi backbone chưa cần gradient).
- Tại epoch `== unfreeze_epoch`:
  1. `model.image_encoder.unfreeze_from(UNFREEZE_FROM)` — mở `layer3` + `layer4` (nếu `UNFREEZE_FROM='layer3'`), giữ `conv1/bn1/layer1/layer2` frozen.
  2. Thêm param group mới vào optimizer với `lr=FINETUNE_LR`.
  3. Tạo lại scheduler (cosine, không warmup).
  4. Giải phóng cache, chuyển sang full forward pass mỗi batch.
- Với `UNFREEZE_EPOCH=0`: cache chỉ dùng để pre-compute 1 lần trước epoch 0, sau đó model fine-tune `layer3+layer4` ngay từ epoch đầu tiên.
- `PROJ_WARMUP_EPOCHS > 0`: chạy thêm pha "chỉ train projection" trên cache **trước** main loop (chỉ có ý nghĩa khi backbone đang frozen, i.e. `unfreeze_epoch != 0`).

---

## 6. Checkpoint registry (`configs/config.py`)

```python
from configs.config import get_save_path, get_best_checkpoint, get_latest_checkpoint, list_checkpoints, update_checkpoint_r1

save_path = get_save_path('cnn_gru', cfg_tag='exp1')   # tự đăng ký vào registry.json
update_checkpoint_r1('cnn_gru', save_path, val_r1=42.3)

best_ckpt = get_best_checkpoint('cnn_gru')              # theo R@1 cao nhất
list_checkpoints('cnn_gru')
```

Tất cả checkpoint + `registry.json` lưu tại `outputs/checkpoints/` (Colab: `/content/drive/MyDrive/project/outputs/checkpoints`).

---

## 7. Kết quả tham khảo

| Model | R@1 (test) |
|-------|-----------|
| CLIP zero-shot (`clip-vit-base-patch32`) | ~54% |
| CNN - GRU | ~23% |
| CNN - LSTM | ~23% |
| CNN - MiniLM | ~ 28%|

Gap chủ yếu do điều kiện train hạn chế (data nhỏ, ít epoch, ResNet50 fine-tune một phần) — *xem chi tiết trong báo cáo (`outputs/results/*_report.txt`) và phân tích lỗi (`*_quantitative.png`, `*_group_analysis.png`, `*_error_analysis.csv`)*.

---

## 8. Yêu cầu môi trường

```
torch, torchvision
transformers (cho MiniLM-L6 / CLIP zero-shot)
pandas, numpy
matplotlib, Pillow
tensorboard
tqdm
```

Chạy tốt trên Google Colab (GPU). `BaseConfig.BASE_DIR` tự động chuyển sang `/content/drive/MyDrive/project` nếu phát hiện Google Drive đã mount, ngược lại dùng path local của project.

Cài đặt:
```python
pip install -r requirements.txt
```
