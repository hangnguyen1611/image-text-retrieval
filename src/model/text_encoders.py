import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


# ----------------------- Attention Pooling -----------------------
class AttentionPool(nn.Module):
    """
    Additive (Bahdanau-style) self-attention pooling.
    Args:
        hidden_dim : chiều của hidden state đầu vào
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, outputs, lengths):
        device = outputs.device
        scores = self.attn(outputs).squeeze(-1)

        # Mask padding
        seq_range = torch.arange(outputs.size(1), device=device)
        pad_mask  = seq_range[None, :] >= lengths.to(device)[:, None]
        scores    = scores.masked_fill(pad_mask, float('-inf'))

        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (outputs * weights).sum(1)


# ----------------------- Base class -----------------------
class _TextEncoderBase(nn.Module):
    """
    Subclass implement extract_features():
      - CNN-based (GRU/LSTM)  : nhận (captions: LongTensor, lengths: LongTensor)
      - Transformer/MiniLM    : nhận (input_ids, attention_mask)
    """
    def extract_features(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        feat = self.projection(self.extract_features(*args, **kwargs))
        return F.normalize(feat, p=2, dim=1) 


# ----------------------- GRU -----------------------
class TextEncoderGRU(_TextEncoderBase):
    """Bidirectional 2-layer GRU + additive attention pooling"""
    def __init__(self, vocab_size, cfg):
        super().__init__()
        embed_dim   = cfg.get('EMBED_DIM', 512)
        hidden_size = cfg.get('HIDDEN_DIM', 512)
        dropout     = cfg.get('DROPOUT', 0.2)

        self.embedding         = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(0.1)
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.attn_pool = AttentionPool(hidden_size * 2)

        self.projection = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, embed_dim),
        )

    def extract_features(self, captions, lengths):
        lengths = lengths.cpu().clamp(min=1)
        embeds  = self.embedding_dropout(self.embedding(captions))
        packed  = nn.utils.rnn.pack_padded_sequence(
            embeds, lengths, batch_first=True, enforce_sorted=False
        )
        outputs, _ = self.gru(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return self.attn_pool(outputs, lengths)


# ----------------------- LSTM -----------------------
class TextEncoderLSTM(_TextEncoderBase):
    """Bidirectional 2-layer LSTM + additive attention pooling"""
    def __init__(self, vocab_size, cfg):
        super().__init__()
        embed_dim   = cfg.get('EMBED_DIM', 512)
        hidden_size = cfg.get('HIDDEN_DIM', 512)
        dropout     = cfg.get('DROPOUT', 0.2)

        self.embedding         = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(0.1)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.attn_pool = AttentionPool(hidden_size * 2)

        self.projection = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, embed_dim),
        )

    def extract_features(self, captions, lengths):
        lengths = lengths.cpu().clamp(min=1)
        embeds  = self.embedding_dropout(self.embedding(captions))
        packed  = nn.utils.rnn.pack_padded_sequence(
            embeds, lengths, batch_first=True, enforce_sorted=False
        )
        outputs, _ = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return self.attn_pool(outputs, lengths)


# ----------------------- MiniLM-L6 -----------------------
MINILM_HF_ID = 'sentence-transformers/all-MiniLM-L6-v2'
MINILM_DIM   = 384   # output dim của all-MiniLM-L6-v2
class TextEncoderMiniLM(_TextEncoderBase):
    """
    Frozen all-MiniLM-L6-v2 (HuggingFace) + trainable projection head.
    Backbone hoàn toàn frozen — chỉ projection được train.
    """
    def __init__(self, cfg: dict):
        super().__init__()
        embed_dim = cfg.get('EMBED_DIM', 512)
        dropout   = cfg.get('DROPOUT', 0.2)

        # ---- Backbone (frozen) ----
        self.backbone = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self._freeze_backbone()

        # ---- Projection (trainable) ----
        self.projection = nn.Sequential(
            nn.LayerNorm(MINILM_DIM),
            nn.Linear(MINILM_DIM, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, embed_dim),
        )

    # ------------------------------------------------------------------
    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        if not any(p.requires_grad for p in self.backbone.parameters()):
            self.backbone.eval()
        return self


    # ------------------------------------------------------------------
    def extract_features(self, input_ids, attention_mask):
        """
        Mean-pooling trên các token không phải padding.
        """
        ctx = torch.no_grad() if not any(p.requires_grad for p in self.backbone.parameters()) else torch.enable_grad()
        with ctx:
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        mask   = attention_mask.unsqueeze(-1).float()
        return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def forward(self, input_ids, attention_mask):
        feat = self.projection(self.extract_features(input_ids, attention_mask))
        return F.normalize(feat, p=2, dim=1)