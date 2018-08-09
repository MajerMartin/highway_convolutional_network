import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


def run_model(model, optimizer, criterion, loader, device, mode, val_size=1):
    """
    Run either model training or inference.

    Args:
        model (nn.Module): instance of the model
        optimizer (torch.optim.<>): instance of the optimizer
        criterion (torch.nn.modules.loss.<>): instance of the loss function
        loader (torch.utils.data.DataLoader): images iterator
        device (torch.device): selected device
        mode (str): "train" for training, "val" or "test" for inference
        val_size (float): percentage of the validation set size

    Returns:
        epoch_loss (float): total epoch loss
        epoch_acc (float): total epoch accuracy
    """
    if mode == "train":
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    running_corrects = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        with torch.set_grad_enabled(mode == "train"):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, targets)
            
            # backward only in training mode
            if mode == "train":
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == targets.data)
    
    # dataset size does not take sampling into account
    data_len = len(loader.dataset)
    if mode == "train":
        data_len *= (1 - val_size)
    elif mode == "val":
        data_len *= val_size

    epoch_loss = running_loss / data_len
    epoch_acc = running_corrects.double() / data_len

    return epoch_loss, epoch_acc


def plot_history(history):
    """
    Plot train and validation losses and accuracies.

    Args:
        history (dict): stored training results
    """
    x = np.arange(1, len(history["train_acc"]) + 1)

    plt.subplot(2, 1, 1)
    plt.plot(x, history["train_loss"])
    plt.plot(x, history["val_loss"])
    plt.ylabel("loss")

    plt.subplot(2, 1, 2)
    plt.plot(x, history["train_acc"])
    plt.plot(x, history["val_acc"])
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend(["train", "validation"], loc="lower right")
    
    plt.show()