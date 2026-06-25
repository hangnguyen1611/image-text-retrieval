import torch.nn as nn


class ContrastiveModel(nn.Module):
    def __init__(self, image_encoder, text_encoder, cfg):
        super().__init__()
        self.image_encoder   = image_encoder
        self.text_encoder    = text_encoder
        self.temperature     = cfg.get('TEMPERATURE', 0.1)
        self.margin          = cfg.get('MARGIN', 0.2)
        self.hard_neg_weight = cfg.get('HARD_NEG_WEIGHT', 0.3)

    def forward(self, images, captions, lengths, attention_mask=None):
        img_emb = self.image_encoder(images)
        if attention_mask is not None:
            txt_emb = self.text_encoder(captions, attention_mask)
        else:
            txt_emb = self.text_encoder(captions, lengths)
        return img_emb, txt_emb