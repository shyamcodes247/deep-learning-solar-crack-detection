# src/model.py
import torch.nn as nn
from torchvision.models import resnet18

def build_model(pretrained=True):
    model = resnet18(weights="IMAGENET1K_V1" if pretrained else None)
    
    # Replace final FC layer: 1000 -> 1 (binary logit)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)
    
    return model