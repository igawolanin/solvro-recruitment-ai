import torch
import numpy as np

def run_test(model, loader, device):
    """
    Run test epoch.

    Parameters:
    - model (torch.nn.Module): Model to be evaluated.
    - loader (DataLoader): DataLoader providing test data.
    - device (torch.device): Device on which computations are performed.

    Returns:
    - predictions (np.ndarray): Predicted class labels.
    - targets (np.ndarray):  True class labels.
    """
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            preds = model(x_batch)
            preds = preds.argmax(1).cpu().numpy()
            predictions.append(preds)
            targets.append(y_batch.cpu().numpy())

    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)

    wrong_indexes = []
    for i in range(len(predictions)):
        if predictions[i] != targets[i]:
            wrong_indexes.append(i)

    return predictions, targets, wrong_indexes