import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import StratifiedShuffleSplit


# statistics computed from the training set
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)

normalize = transforms.Normalize(MNIST_MEAN, MNIST_STD)


def get_train_and_val_loader(data_dir, batch_size, random_seed,
                             augmentation=True, val_size=0.3,
                             num_workers=1, pin_memory=False):
    """
    Create loaders using tratified train and validation split
    for the MNIST dataset.

    Args:
        data_dir (str): dataset directory
        batch_size (int): samples count per batch
        random_seed (int): seed for the random number generator
        augmentation (bool): augment the training set
        val_size (float): percentage of the validation set size
        num_workers (int): number of workers, 1 if using CUDA
        pin_memory (bool): copy tensors to CUDA memory, True if using CUDA

    Returns:
        train_loader (torch.utils.data.DataLoader): training images iterator
        val_loader (torch.utils.data.DataLoader): validation images iterator
    """
    if augmentation:
        train_transform = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    train_dataset = datasets.MNIST(root=data_dir, transform=train_transform,
                                   train=True, download=True)

    val_dataset = datasets.MNIST(root=data_dir, transform=val_transform,
                                 train=True, download=True)

    # collect all the labels for stratified shuffle split
    data_len = len(train_dataset)
    labels = [train_dataset[i][1].numpy().tolist() for i in range(data_len)]
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size,
                                 random_state=random_seed)

    train_index, val_index = next(sss.split(np.zeros(data_len), labels))
    
    # create sampled dataloaders
    train_sampler = SubsetRandomSampler(train_index)
    val_sampler = SubsetRandomSampler(val_index)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                              num_workers=num_workers, pin_memory=pin_memory)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler,
                            num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader


def get_test_loader(data_dir, batch_size, num_workers=1, pin_memory=False):
    """
    Create loader for the testing image for the MNIST dataset.

    Args:
        data_dir (str): dataset directory
        batch_size (int): samples count per batch
        num_workers (int): number of workers, 1 if using CUDA
        pin_memory (bool): copy tensors to CUDA memory, True if using CUDA

    Returns:
        test_loader (torch.utils.data.DataLoader): testing images iterator
    """
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    test_dataset = datasets.MNIST(root=data_dir, transform=test_transform,
                                  train=False, download=True)

    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             num_workers=num_workers, pin_memory=pin_memory)

    return test_loader