import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class ImageEncoder(nn.Module):
    _LAYER_IDX = {'layer1': 4, 'layer2': 5, 'layer3': 6, 'layer4': 7}

    def __init__(self, cfg):
        super().__init__()
        resnet        = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool  = nn.AdaptiveAvgPool2d((1, 1))

        embed_dim = cfg.get('EMBED_DIM', 512)
        dropout   = cfg.get('DROPOUT', 0.3)

        if cfg.get('FREEZE_BACKBONE', True):
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.projection = nn.Sequential(
            nn.LayerNorm(2048),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, embed_dim),
            nn.Dropout(dropout * 0.5),
        )

    def unfreeze_from(self, from_layer: str = 'layer4'):
        """
        Unfreeze từ from_layer trở đi.
        from_layer: 'layer1' | 'layer2' | 'layer3' | 'layer4'
        """
        start_idx = self._LAYER_IDX[from_layer]
        for idx, module in enumerate(self.backbone):
            if idx >= start_idx:
                for param in module.parameters():
                    param.requires_grad = True

    def unfreeze_layer4(self):
        self.unfreeze_from('layer4')

    @property
    def is_frozen(self):
        return not any(p.requires_grad for p in self.backbone.parameters())

    def extract_features(self, images):
        feats = self.backbone(images)
        feats = self.avgpool(feats)
        return torch.flatten(feats, 1)

    def forward(self, images):
        feat = self.avgpool(self.backbone(images)).flatten(1)
        return F.normalize(self.projection(feat), p=2, dim=1)