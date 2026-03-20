import torch

def compute_cam(model, img, label, device):
    """
    Compute CAM for given image and target class.

    Extracts feature maps from the last convolutional layer and combines them
    with the weights of the final fully connected layer to highlight regions
    important for the selected class.

    Parameters:
    - model (torch.nn.Module): Trained model.
    - img (torch.Tensor): Input image tensor of shape (C, H, W).
    - label (int): Target class index for which CAM is computed.
    - device (torch.device): Device on which computations are performed.

    Returns:
    - cam (torch.Tensor): Raw CAM of shape (H, W), representing spatial importance.
    """
    model.eval()
    blobs = []

    def hook(module, input, output):
        blobs.append(output.detach())

    img = torch.unsqueeze(img, 0)
    img = img.to(device)
    handle = model.layer4.register_forward_hook(hook)
    model(img)
    handle.remove()
    blobs = blobs[0].squeeze()
    weights = model.fc.weight[int(label)]
    cam = torch.sum(weights.unsqueeze(-1).unsqueeze(-1) * blobs, dim=0)

    return cam