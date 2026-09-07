"""
MedCXRNet Model Architecture Implementation.

Custom 4-stage Convolutional Neural Network trained from scratch for 14-class multi-label
Chest X-Ray pathology classification. Designed for 224x224 grayscale input and low VRAM footprint.
"""

from typing import Dict
import torch
import torch.nn as nn
from configs.config import NUM_CLASSES, NUM_CHANNELS


class ConvBlock(nn.Module):
    """
    Standard Convolutional Block: Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MedCXRNet(nn.Module):
    """
    MedCXRNet Architecture:
        - Input: (Batch, 1, 224, 224)
        - Stage 1: Conv2d(1 -> 32) -> BatchNorm -> ReLU -> MaxPool (112x112)
        - Stage 2: Conv2d(32 -> 64) -> BatchNorm -> ReLU -> MaxPool (56x56)
        - Stage 3: Conv2d(64 -> 128) -> BatchNorm -> ReLU -> MaxPool (28x28)
        - Stage 4: Conv2d(128 -> 256) -> BatchNorm -> ReLU -> MaxPool (14x14)
        - Head: AdaptiveAvgPool2d(1, 1) -> Flatten -> Dropout(0.30) -> Linear(256 -> 14)
        - Output: 14 raw unnormalized logits (No Sigmoid layer inside)
    """

    def __init__(self, in_channels: int = NUM_CHANNELS, num_classes: int = NUM_CLASSES, dropout_rate: float = 0.30):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # 4 Convolutional Blocks
        self.block1 = ConvBlock(in_channels, 32)
        self.block2 = ConvBlock(32, 64)
        self.block3 = ConvBlock(64, 128)
        self.block4 = ConvBlock(128, 256)

        # Global Average Pooling and Head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(256, num_classes)

        # Initialize weights using Kaiming Normal (He Init)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        Apply Kaiming Normal (He) initialization to Convolutional layers
        and constant initialization to BatchNorm layers for robust scratch convergence.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input image tensor of shape (Batch, 1, 224, 224).

        Returns:
            torch.Tensor: Unnormalized raw logits of shape (Batch, 14).
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count total and trainable parameters of a PyTorch model.

    Args:
        model (nn.Module): PyTorch model instance.

    Returns:
        Dict[str, int]: Dictionary containing 'total_parameters' and 'trainable_parameters'.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable
    }
