from torchvision import models
import torch.nn as nn

def get_resnet18(output_n, freeze_backbone=True):
    """
    Initialize a ResNet18 model with optional frozen backbone.

    Parameters:
    - output_n (int): Number of output classes.
    - freeze_backbone (bool, optional):
        If True, all pretrained layers are frozen and only the final
        classification layer (fc) is trainable.
        If False, the entire model is trainable.

    Returns:
    - model (torch.nn.Module): ResNet18 model ready for training.
    """
    model = models.resnet18(weights='IMAGENET1K_V1')
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, output_n)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
    return model