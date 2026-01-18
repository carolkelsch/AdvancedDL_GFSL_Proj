import torch
import torch.nn as nn
import torchvision.models as models

def get_backbone(name="resnet50"):
    if name == "resnet50":
        model = models.resnet50(weights="IMAGENET1K_V1")
        feat_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, feat_dim

    elif name == "resnet18":
        model = models.resnet18()
        feat_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, feat_dim

    else:
        raise ValueError(f"Unknown backbone: {name}")

class EmbeddingModel(nn.Module):
    def __init__(self, backbone_name="resnet50", embed_dim=128):
        super().__init__()
        self.backbone, feat_dim = get_backbone(backbone_name)
        self.embedding = nn.Linear(feat_dim, embed_dim)
        '''self.embedding = nn.Sequential(
            nn.Linear(feat_dim, embed_dim),
            nn.BatchNorm1d(embed_dim)
        )'''

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.embedding(features)
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings