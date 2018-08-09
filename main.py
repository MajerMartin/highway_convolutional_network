import torch
import torch.nn as nn
import torch.optim as optim

from model import HighwayCNN
from utilities import run_model, plot_history
from data_loaders import get_train_and_val_loader, get_test_loader


# define the parameters for data loaders
# TODO: use argparse
SEED = 42
DATA_DIR = "data"
VAL_SIZE = 0.3
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 512
EPOCHS = 15


# Windows needs this because of multiprocessing
def main():
    torch.manual_seed(SEED)

    # set up the device parameters
    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda:0" if use_cuda else "cpu")
    cuda_kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    # load the data
    train_loader, val_loader = get_train_and_val_loader(DATA_DIR,
                                                        TRAIN_BATCH_SIZE,
                                                        SEED,
                                                        augmentation=True,
                                                        val_size=VAL_SIZE,
                                                        **cuda_kwargs)

    test_loader = get_test_loader(DATA_DIR, TEST_BATCH_SIZE, **cuda_kwargs)

    # initialize the model, the optimizer, the scheduler and the criterion
    model = HighwayCNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01,
                          momentum=0.9, nesterov=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)
    criterion = nn.CrossEntropyLoss()

    # train and test the model
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_model(model, optimizer, criterion, 
                                          train_loader, device, "train",
                                          val_size=VAL_SIZE)
        val_loss, val_acc = run_model(model, optimizer, criterion,
                                      val_loader, device, "val",
                                      val_size=VAL_SIZE)
        
        print(" ".join(["epoch: {} |",
                        "train_loss: {:.4f} |",
                        "train_acc: {:.3f} |",
                        "val_loss: {:.4f} |",
                        "val_acc: {:.3f}"]).format(epoch,train_loss,
                                                   train_acc, val_loss,
                                                   val_acc))

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

    test_acc = run_model(model, optimizer, criterion, test_loader,
                         device, "test", val_size=VAL_SIZE)[1]

    print("\nAccuracy on testing set: {:.3f}".format(test_acc))
    plot_history(history)


if __name__ == "__main__":
    main()