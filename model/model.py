import torch
from torch import nn
from torchvision import models

num_classes = 2

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device = get_device()

model = model.to(device)
