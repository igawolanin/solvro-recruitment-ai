import torch

def run_one_epoch(loader, model, criterion, optimizer, device):
    """
    Run one training epoch.

    Performs forward pass, loss computation, backpropagation and optimizer step
    for all batches in the training dataset.

    Parameters:
    - loader (DataLoader): DataLoader for training data.
    - model (torch.nn.Module): Model to be trained.
    - criterion (callable): Loss function.
    - optimizer (torch.optim.Optimizer): Optimizer used for updating model parameters.
    - device (torch.device): Device on which computations are performed.

    Returns:
    - epoch_loss (float): Average loss over the epoch.
    - accuracy (float): Accuracy over the epoch.
    """
    epoch_loss, correct, total = 0,0,0
    model.train()
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        predictions = model(x_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * x_batch.size(0)
        preds = predictions.argmax(1)
        correct += (preds==y_batch).sum().item()
        total += y_batch.size(0)
    return epoch_loss / total, correct / total

def run_val_epoch(loader, model, criterion, device):
    """
    Run one validation epoch.

    Evaluates the model on validation data without updating weights.
    Gradients are disabled to improve performance.

    Parameters:
    - loader (DataLoader): DataLoader for validation data.
    - model (torch.nn.Module): Model to be evaluated.
    - criterion (callable): Loss function.
    - device (torch.device): Device on which computations are performed.

    Returns:
    - epoch_loss (float): Average validation loss.
    - accuracy (float): Validation accuracy.
    """
    epoch_loss, correct, total = 0,0,0
    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            predictions = model(x_batch)
            loss = criterion(predictions, y_batch)
            epoch_loss += loss.item() * x_batch.size(0)
            preds = predictions.argmax(1)
            correct += (preds==y_batch).sum().item()
            total += y_batch.size(0)
    return epoch_loss / total, correct / total

def run_epochs(epochs, train_loader, val_loader, model, optimizer, criterion, device):
    """
    Train and evaluate model over multiple epochs.

    Runs training and validation loops for a specified number of epochs,
    collecting loss and accuracy metrics for both sets.

    Parameters:
    - epochs (int): Number of training epochs.
    - train_loader (DataLoader): DataLoader for training data.
    - val_loader (DataLoader): DataLoader for validation data.
    - model (torch.nn.Module): Model to be trained.
    - optimizer (torch.optim.Optimizer): Optimizer used for updating model parameters.
    - criterion (callable): Loss function.
    - device (torch.device): Device on which computations are performed.

    Returns:
    - history (dict): Dictionary containing:
        - train_loss (list of float): Training loss per epoch.
        - train_acc (list of float): Training accuracy per epoch.
        - val_loss (list of float): Validation loss per epoch.
        - val_acc (list of float): Validation accuracy per epoch.
    """
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss":   [],
        "val_acc":   []
    }
    for epoch in range(epochs):
        train_loss, train_accuracy = run_one_epoch(train_loader, model, criterion, optimizer, device)
        val_loss, val_accuracy = run_val_epoch(val_loader, model, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_accuracy)

        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_accuracy)
        print(f"Epoch {epoch+1}/{epochs} | train loss: {train_loss:.4f}, acc: {train_accuracy:.4f} | val loss: {val_loss:.4f}, acc: {val_accuracy:.4f}")

    return history
