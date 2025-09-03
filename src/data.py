import math
import torch
import torch.utils.data
from pathlib import Path
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import multiprocessing

from .helpers import compute_mean_and_std, get_data_location


def get_data_loaders(batch_size: int = 32, valid_size: float = 0.2, num_workers: int = -1, limit: int = -1):
    """
    Orchestrates the creation of training, validation, and test data loaders.

    This function sets up the entire data loading and preprocessing pipeline. It defines
    distinct data augmentation and normalization transformations for the training set (to
    improve model generalization) and the validation/test sets (for consistent
    evaluation). It loads images from disk, splits the training data into training and
    validation subsets, and wraps them in `DataLoader` objects for efficient batching
    and parallel loading.

    Args:
        batch_size (int): The number of samples per batch.
        valid_size (float): The fraction of the training data to reserve for validation.
        num_workers (int): The number of subprocesses to use for data loading. -1 means
                         using all available CPUs.
        limit (int): The maximum number of data points to use. Useful for debugging.

    Returns:
        dict: A dictionary containing 'train', 'valid', and 'test' `DataLoader` objects.
    """

    if num_workers == -1:
        # Use all cores
        num_workers = multiprocessing.cpu_count()

    # We will fill this up later
    data_loaders = {"train": None, "valid": None, "test": None}

    base_path = Path(get_data_location())

    # Compute mean and std of the dataset
    mean, std = compute_mean_and_std()
    print(f"Dataset mean: {mean}, std: {std}")

    """
    The following dictionary defines the transformations for the training, validation,
    and testing datasets.

    - The 'train' transform includes data augmentation (RandomResizedCrop,
      RandomHorizontalFlip, RandomRotation, ColorJitter) to help the model generalize
      and prevent overfitting.
    - The 'valid' and 'test' transforms are identical and use a deterministic
      CenterCrop. This ensures that we get a consistent, comparable evaluation of
      the model's performance on unseen data.
    """
    data_transforms = {
        "train": transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        "valid": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        "test": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    }

    # Create train and validation datasets
    train_data = datasets.ImageFolder(
        base_path / "train",
        transform=data_transforms["train"]
    )

    valid_data = datasets.ImageFolder(
        base_path / "train",
        transform=data_transforms["valid"]
    )

    # obtain training indices that will be used for validation
    n_tot = len(train_data)
    indices = torch.randperm(n_tot)

    # If requested, limit the number of data points to consider
    if limit > 0:
        indices = indices[:limit]
        n_tot = limit

    split = int(math.ceil(valid_size * n_tot))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)

    # prepare data loaders
    data_loaders["train"] = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers
    )

    data_loaders["valid"] = torch.utils.data.DataLoader(
        valid_data,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=num_workers
    )

    # Create test dataset
    test_data = datasets.ImageFolder(
        base_path / "test",
        transform=data_transforms["test"]
    )

    if limit > 0:
        indices = torch.arange(limit)
        test_sampler = torch.utils.data.SubsetRandomSampler(indices)
    else:
        test_sampler = None

    # Create test dataloader
    data_loaders["test"] = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        sampler=test_sampler,
        shuffle=False,
        num_workers=num_workers
    )

    return data_loaders


def visualize_one_batch(data_loaders, max_n: int = 5):
    """
    Fetches and displays a single batch of images from the training data loader.

    This utility function is useful for visually inspecting the data augmentation
    and preprocessing pipeline. It takes a batch of images, reverses the normalization
    transformation to make them viewable, and plots them along with their
    corresponding class labels.

    Args:
        data_loaders (dict): A dictionary of `DataLoader` objects, expected to have
                           at least a 'train' key.
        max_n (int): The maximum number of images to display from the batch.
    """

    # obtain one batch of training images
    # First obtain an iterator from the train dataloader
    # obtain one batch of training images
    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    # Undo the normalization (for visualization purposes)
    mean, std = compute_mean_and_std()
    invTrans = transforms.Compose(
        [
            transforms.Normalize(mean=[0.0, 0.0, 0.0], std=1 / std),
            transforms.Normalize(mean=-mean, std=[1.0, 1.0, 1.0]),
        ]
    )

    images = invTrans(images)

    # Get class names from the train data loader
    class_names = data_loaders["train"].dataset.classes  # Third YOUR CODE HERE

    # Convert from BGR (the format used by pytorch) to
    # RGB (the format expected by matplotlib)
    images = torch.permute(images, (0, 2, 3, 1)).clip(0, 1)

    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(25, 4))
    for idx in range(max_n):
        ax = fig.add_subplot(1, max_n, idx + 1, xticks=[], yticks=[])
        ax.imshow(images[idx])
        # print out the correct label for each image
        # .item() gets the value contained in a Tensor
        ax.set_title(class_names[labels[idx].item()])


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    return get_data_loaders(batch_size=2, num_workers=0)


def test_data_loaders_keys(data_loaders):
    assert set(data_loaders.keys()) == {"train", "valid",
                                        "test"}, "The keys of the data_loaders dictionary should be train, valid and test"


def test_data_loaders_output_type(data_loaders):
    # Test the data loaders

    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)  # Changed from dataiter.next()

    assert isinstance(images, torch.Tensor), "images should be a Tensor"
    assert isinstance(labels, torch.Tensor), "labels should be a Tensor"
    assert images[0].shape[-1] == 224, "The tensors returned by your dataloaders should be 224x224. Did you " \
                                       "forget to resize and/or crop?"


def test_data_loaders_output_shape(data_loaders):
    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    assert len(images) == 2, f"Expected a batch of size 2, got size {len(images)}"
    assert (
            len(labels) == 2
    ), f"Expected a labels tensor of size 2, got size {len(labels)}"


def test_visualize_one_batch(data_loaders):
    visualize_one_batch(data_loaders, max_n=2)
