"""Helpers module for flower recognition deep learning project.

This module provides utility functions for data loading, preprocessing, training,
validation, and testing of deep learning models for flower image classification.
It includes functions for computing dataset statistics, creating data loaders with
augmentation, training loops, and evaluation metrics.

Author: Udacity Deep Learning Nanodegree
Project: Landmark Classification CNN
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
from tqdm import tqdm
import multiprocessing
from pathlib import Path
import os
from torchvision import transforms as T
import math
import random

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def get_device():
    """Determine the best available device for PyTorch computations.
    
    This function checks for available compute devices in order of preference:
    MPS (Apple Silicon GPU) > CUDA (NVIDIA GPU) > CPU. It returns the appropriate
    torch.device object and prints which device is being used.
    
    Returns:
        torch.device: The best available device for computation.
        
    Note:
        - MPS is preferred on Apple Silicon Macs for GPU acceleration
        - CUDA is used if available on systems with NVIDIA GPUs
        - Falls back to CPU if neither MPS nor CUDA are available
        - Prints device selection for user feedback
        
    Example:
        >>> device = get_device()
        Using MPS (Metal Performance Shaders)
        >>> model = model.to(device)
        >>> data = data.to(device)
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    return device


def get_data_location():
    """Find the location of the dataset, either locally or in the Udacity workspace.
    
    This function searches for the flowers dataset in common locations and returns
    the path to the dataset folder. It checks for the dataset in the current directory
    first, then in the lectures subdirectory structure.
    
    Returns:
        str: Path to the dataset folder containing the flower images.
        
    Raises:
        IOError: If the dataset folder is not found in any of the expected locations.
        
    Example:
        >>> data_path = get_data_location()
        >>> print(f"Dataset found at: {data_path}")
        Dataset found at: flowers
    """

    if os.path.exists("flowers"):
        data_folder = "flowers"
    elif os.path.exists("lectures/recognized_flowers/flowers"):
        data_folder = "lectures/recognized_flowers/flowers"
    else:
        raise IOError("Please download the dataset first")

    return data_folder


# Compute image normalization
def compute_mean_and_std():
    """Compute per-channel mean and standard deviation of the dataset.
    
    This function calculates the mean and standard deviation for each color channel
    (RGB) across the entire dataset. These statistics are used for data normalization
    in the transforms.Normalize() function. The computation is cached to avoid
    recalculation on subsequent runs.
    
    The function performs a two-pass algorithm:
    1. First pass: Calculate the mean across all images
    2. Second pass: Calculate the variance using the computed mean
    
    Returns:
        tuple: A tuple containing:
            - mean (torch.Tensor): Per-channel mean values [R, G, B]
            - std (torch.Tensor): Per-channel standard deviation values [R, G, B]
            
    Note:
        Results are cached in 'mean_and_std.pt' file to avoid recomputation.
        The function uses all available CPU cores for faster processing.
        
    Example:
        >>> mean, std = compute_mean_and_std()
        >>> print(f"Mean: {mean}, Std: {std}")
        Mean: tensor([0.485, 0.456, 0.406]), Std: tensor([0.229, 0.224, 0.225])
    """

    cache_file = "mean_and_std.pt"
    if os.path.exists(cache_file):
        print(f"Reusing cached mean and std")
        d = torch.load(cache_file)

        return d["mean"], d["std"]

    folder = get_data_location()
    ds = datasets.ImageFolder(
        folder, transform=T.Compose([T.ToTensor()])
    )
    dl = torch.utils.data.DataLoader(
        ds, batch_size=1, num_workers=multiprocessing.cpu_count()
    )

    mean = 0.0
    # First pass: Compute per-channel mean across entire dataset
    # This loop processes each image batch and accumulates channel-wise means
    for images, _ in tqdm(dl, total=len(ds), desc="Computing mean", ncols=80):
        """Mean computation loop.
        
        For each batch of images:
        1. Get batch size (usually 1 since batch_size=1 in DataLoader)
        2. Reshape images from [batch, channels, height, width] to [batch, channels, pixels]
        3. Compute mean across spatial dimensions (axis=2) for each channel
           - axis=2 refers to the pixel dimension after reshaping
           - Original: [batch=1, channels=3, height=224, width=224]
           - Reshaped: [batch=1, channels=3, pixels=50176] 
           - images.mean(2) computes mean across all 50176 pixels for each channel
        4. Sum across batch dimension (axis=0) to accumulate channel means
           - axis=0 refers to the batch dimension
           - .sum(0) sums across batches to get total for each channel [R, G, B]
           - Since batch_size=1, this just removes the batch dimension

        TENSOR DIMENSION BREAKDOWN:
        ---------------------------
        Original tensor shape: [batch, channels, height, width] = [1, 3, 224, 224]
        - Dimension 0 (axis=0): batch dimension (1 image)
        - Dimension 1 (axis=1): channel dimension (3 channels: R, G, B)
        - Dimension 2 (axis=2): height dimension (224 pixels)
        - Dimension 3 (axis=3): width dimension (224 pixels)
        
        After reshaping with .view(batch_samples, images.size(1), -1):
        Reshaped tensor shape: [batch, channels, flattened_pixels] = [1, 3, 50176]
        - Dimension 0 (axis=0): batch dimension (still 1 image)
        - Dimension 1 (axis=1): channel dimension (still 3 channels: R, G, B)  
        - Dimension 2 (axis=2): flattened pixel dimension (224×224 = 50176 pixels)
        
        KEY CONCEPT: Even after flattening, the tensor still has 3 dimensions!
        The pixels are flattened but they occupy dimension 2 of the tensor.
        
        OPERATIONS EXPLAINED:
        ---------------------
        1. images.mean(2): Compute mean along axis=2 (the 50176 flattened pixels)
           - Input:  [1, 3, 50176] 
           - Output: [1, 3] (mean pixel value for each channel in each batch)
           
        2. .sum(0): Sum along axis=0 (the batch dimension)
           - Input:  [1, 3] 
           - Output: [3] (accumulated channel means: [R_mean, G_mean, B_mean])
           - Since batch=1, this just removes the batch dimension
        
        Variables:
            batch_samples (int): Number of images in current batch (1)
            images (torch.Tensor): Reshaped tensor [1, 3, 50176]
            mean (torch.Tensor): Accumulated channel means [R_total, G_total, B_total]
        """
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
    mean = mean / len(dl.dataset)  # Normalize by total number of images

    var = 0.0
    npix = 0
    # Second pass: Compute per-channel variance using the computed mean
    # This loop calculates squared deviations from mean for standard deviation
    for images, _ in tqdm(dl, total=len(ds), desc="Computing std", ncols=80):
        """Variance computation loop.
        
        For each batch of images:
        1. Get batch size and reshape images to [batch, channels, pixels]
        2. Broadcast mean to match image dimensions: [channels] -> [channels, 1]
        3. Compute squared differences: (pixel_value - channel_mean)²
        4. Sum across batch and spatial dimensions to accumulate variance
        5. Count total pixels for final normalization
        
        Variables:
            batch_samples (int): Number of images in current batch
            images (torch.Tensor): Reshaped tensor [batch, channels, height*width]
            mean.unsqueeze(1) (torch.Tensor): Mean broadcasted to [channels, 1]
            var (torch.Tensor): Accumulated sum of squared deviations per channel
            npix (int): Total number of pixels processed across all images
        
        Mathematical operation:
            var += Σ(pixel - μ)² for each channel across batch and spatial dims
        """
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1)) ** 2).sum([0, 2])
        npix += images.nelement()  # Total pixels = batch_size * channels * height * width

    std = torch.sqrt(var / (npix / 3))

    # Cache results so we don't need to redo the computation
    torch.save({"mean": mean, "std": std}, cache_file)

    return mean, std


def get_transforms(rand_augment_magnitude):
    """Create data transformation pipelines for training and validation.
    
    This function creates separate transformation pipelines for training and validation
    datasets. The training pipeline includes data augmentation techniques like random
    cropping, horizontal flipping, and RandAugment, while the validation pipeline
    only includes basic preprocessing without augmentation.
    
    Args:
        rand_augment_magnitude (int): Magnitude parameter for RandAugment transformation.
            Higher values apply stronger augmentations. Typical range is 0-30.
            
    Returns:
        dict: Dictionary containing transformation pipelines with keys:
            - 'train': Training transformations with augmentation
            - 'valid': Validation transformations without augmentation
            
    Note:
        All transformations normalize images using dataset-specific mean and std.
        Images are resized to 256x256 then cropped to 224x224 for model input.
        
    Example:
        >>> transforms = get_transforms(rand_augment_magnitude=9)
        >>> train_transform = transforms['train']
        >>> valid_transform = transforms['valid']
    """

    # These are the per-channel mean and std of CIFAR-10 over the dataset
    mean, std = compute_mean_and_std()

    # Define our transformations
    return {
        "train": T.Compose(
            [
                # All images in CIFAR-10 are 32x32. We enlarge them a bit so we can then
                # take a random crop
                T.Resize(256),
                
                # take a random part of the image
                T.RandomCrop(224),
                
                # Horizontal flip is not part of RandAugment according to the RandAugment
                # paper
                T.RandomHorizontalFlip(0.5),
                
                # RandAugment has 2 main parameters: how many transformations should be
                # applied to each image, and the strength of these transformations. This
                # latter parameter should be tuned through experiments: the higher the more
                # the regularization effect
                T.RandAugment(
                    num_ops=2,
                    magnitude=rand_augment_magnitude,
                    interpolation=T.InterpolationMode.BILINEAR,
                ),
                T.ToTensor(),
                T.Normalize(mean, std),
            ]
        ),
        "valid": T.Compose(
            [
                # Both of these are useless, but we keep them because
                # in a non-academic dataset you will need them
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean, std),
            ]
        )
    }


def get_data_loaders(
    batch_size: int = 32, valid_size: float = 0.2, num_workers: int = -1, limit: int = -1, rand_augment_magnitude: int = 9
):
    """Create and return training and validation data loaders.
    
    This function creates PyTorch DataLoader objects for training and validation
    datasets with appropriate transformations and sampling strategies. The function
    automatically splits the dataset into training and validation sets based on
    the specified validation size.
    
    Args:
        batch_size (int, optional): Size of mini-batches for training. Defaults to 32.
        valid_size (float, optional): Fraction of dataset to use for validation.
            Must be between 0 and 1. For example, 0.2 means 20% for validation.
            Defaults to 0.2.
        num_workers (int, optional): Number of worker processes for data loading.
            Use -1 to utilize all available CPU cores. Defaults to -1.
        limit (int, optional): Maximum number of data points to consider.
            Use -1 for no limit. Useful for debugging with smaller datasets.
            Defaults to -1.
        rand_augment_magnitude (int, optional): Magnitude for RandAugment transformations.
            Higher values apply stronger augmentations. Defaults to 9.
            
    Returns:
        dict: Dictionary containing data loaders with keys:
            - 'train': Training data loader with augmented data
            - 'valid': Validation data loader without augmentation
            
    Note:
        The function prints dataset statistics (mean and std) for verification.
        Random sampling ensures different train/validation splits across runs.
        
    Example:
        >>> loaders = get_data_loaders(batch_size=64, valid_size=0.15)
        >>> train_loader = loaders['train']
        >>> valid_loader = loaders['valid']
        >>> print(f"Training batches: {len(train_loader)}")
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

    # Get the transforms for training and validation
    data_transforms = get_transforms(rand_augment_magnitude)

    # Create train and validation datasets
    train_data = datasets.ImageFolder(
        base_path,
        # add the appropriate transform that you defined in
        # the data_transforms dictionary
        transform=data_transforms["train"]
    )
    # The validation dataset is a split from the train_one_epoch dataset, so we read
    # from the same folder, but we apply the transforms for validation
    valid_data = datasets.ImageFolder(
        base_path,
        # add the appropriate transform that you defined in
        # the data_transforms dictionary
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
    valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)  # =

    # prepare data loaders
    data_loaders["train"] = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
    )
    data_loaders["valid"] = torch.utils.data.DataLoader(
        valid_data,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=num_workers,
    )

    return data_loaders


def train_one_epoch(train_dataloader, model, optimizer, loss):
    """Perform one epoch of model training.
    
    This function executes a complete training epoch, processing all batches
    in the training dataset. It handles the forward pass, loss computation,
    backpropagation, and parameter updates for each batch.
    
    Args:
        train_dataloader (torch.utils.data.DataLoader): DataLoader containing
            training data batches.
        model (torch.nn.Module): Neural network model to train.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        loss (torch.nn.Module): Loss function for computing training loss.
        
    Returns:
        float: Average training loss across all batches in the epoch.
        
    Note:
        - Automatically moves model and data to best available device (MPS/CUDA/CPU)
        - Sets model to training mode for proper batch normalization and dropout
        - Uses running average to compute epoch loss
        - Displays progress bar with tqdm
        
    Example:
        >>> train_loss = train_one_epoch(train_loader, model, optimizer, criterion)
        >>> print(f"Training loss: {train_loss:.4f}")
    """

    # Get the best available device and move model to it
    device = get_device()
    model = model.to(device)

    # Set the model in training mode
    # (so all layers that behave differently between training and evaluation,
    # like batchnorm and dropout, will select their training behavior)
    model.train()  # -

    # Loop over the training data
    train_loss = 0.0

    for batch_idx, (data, target) in tqdm(
        enumerate(train_dataloader),
        desc="Training",
        total=len(train_dataloader),
        leave=True,
        ncols=80,
    ):
        # move data to the selected device
        data, target = data.to(device), target.to(device)

        # 1. clear the gradients of all optimized variables
        optimizer.zero_grad()  # -
        # 2. forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)  # =
        # 3. calculate the loss
        loss_value = loss(output, target)  # =
        # 4. backward pass: compute gradient of the loss with respect to model parameters
        loss_value.backward()  # -
        # 5. perform a single optimization step (parameter update)
        optimizer.step()  # -

        # update average training loss
        train_loss = train_loss + (
            (1 / (batch_idx + 1)) * (loss_value.data.item() - train_loss)
        )

    return train_loss


def valid_one_epoch(valid_dataloader, model, loss):
    """Perform one epoch of model validation.
    
    This function evaluates the model on the validation dataset without updating
    model parameters. It computes the validation loss to monitor model performance
    and detect overfitting during training.
    
    Args:
        valid_dataloader (torch.utils.data.DataLoader): DataLoader containing
            validation data batches.
        model (torch.nn.Module): Neural network model to validate.
        loss (torch.nn.Module): Loss function for computing validation loss.
        
    Returns:
        float: Average validation loss across all batches.
        
    Note:
        - Disables gradient computation for efficiency using torch.no_grad()
        - Sets model to evaluation mode for proper batch normalization and dropout
        - Automatically moves model and data to best available device (MPS/CUDA/CPU)
        - Uses running average to compute epoch loss
        - Displays progress bar with tqdm
        
    Example:
        >>> valid_loss = valid_one_epoch(valid_loader, model, criterion)
        >>> print(f"Validation loss: {valid_loss:.4f}")
    """

    # Get the best available device
    device = get_device()
    
    # During validation we don't need to accumulate gradients
    with torch.no_grad():

        # set the model to evaluation mode
        # (so all layers that behave differently between training and evaluation,
        # like batchnorm and dropout, will select their evaluation behavior)
        model.eval()  # -

        # Move the model to the selected device
        model = model.to(device)

        # Loop over the validation dataset and accumulate the loss
        valid_loss = 0.0
        for batch_idx, (data, target) in tqdm(
            enumerate(valid_dataloader),
            desc="Validating",
            total=len(valid_dataloader),
            leave=True,
            ncols=80,
        ):
            # move data to the selected device
            data, target = data.to(device), target.to(device)

            # 1. forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)  # =
            # 2. calculate the loss
            loss_value = loss(output, target)  # =

            # Calculate average validation loss
            valid_loss = valid_loss + (
                (1 / (batch_idx + 1)) * (loss_value.data.item() - valid_loss)
            )

    return valid_loss


def optimize(data_loaders, model, optimizer, loss, n_epochs, save_path, interactive_tracking=False):
    """Train and validate model over multiple epochs with optimization.
    
    This function orchestrates the complete training process, including training
    and validation loops, learning rate scheduling, model checkpointing, and
    optional interactive loss plotting. It implements early stopping based on
    validation loss improvement.
    
    Args:
        data_loaders (dict): Dictionary containing 'train' and 'valid' DataLoaders.
        model (torch.nn.Module): Neural network model to train.
        optimizer (torch.optim.Optimizer): Optimizer for parameter updates.
        loss (torch.nn.Module): Loss function for training and validation.
        n_epochs (int): Number of training epochs to run.
        save_path (str): File path to save the best model weights.
        interactive_tracking (bool, optional): Whether to display live loss plots.
            Defaults to False.
            
    Returns:
        None: Function saves best model to save_path and optionally displays plots.
        
    Note:
        - Uses ReduceLROnPlateau scheduler to reduce learning rate when validation
          loss plateaus
        - Saves model weights when validation loss improves by more than 1%
        - Tracks training loss, validation loss, and learning rate
        - Interactive tracking requires matplotlib backend support
        
    Example:
        >>> optimize(data_loaders, model, optimizer, criterion, 
        ...          n_epochs=50, save_path='best_model.pth', 
        ...          interactive_tracking=True)
    """
    
    def after_subplot(ax: plt.Axes, group_name: str, x_label: str):
        """Add title xlabel and legend to single chart"""
        ax.set_title(group_name)
        ax.set_xlabel(x_label)
        ax.legend(loc="center right")
        
    # initialize tracker for minimum validation loss
    if interactive_tracking:
        liveloss = PlotLosses(outputs=[MatplotlibPlot(after_subplot=after_subplot)])
    else:
        liveloss = None

    valid_loss_min = None
    logs = {}

    # Learning rate scheduler: setup a learning rate scheduler that
    # reduces the learning rate when the validation loss reaches a
    # plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(  # =
        optimizer, "min", verbose=True, threshold=0.01  # -
    )  

    for epoch in range(1, n_epochs + 1):

        train_loss = train_one_epoch(
            data_loaders["train"], model, optimizer, loss
        )

        valid_loss = valid_one_epoch(data_loaders["valid"], model, loss)

        # If the validation loss decreases by more than 1%, save the model
        if valid_loss_min is None or (
                (valid_loss_min - valid_loss) / valid_loss_min > 0.01
        ):

            # Save the weights to save_path
            torch.save(model.state_dict(), save_path)  # -

            valid_loss_min = valid_loss

        # Update learning rate, i.e., make a step in the learning rate scheduler
        scheduler.step(valid_loss)

        # Log the losses and the current learning rate
        if interactive_tracking:
            logs["loss"] = train_loss
            logs["val_loss"] = valid_loss
            logs["lr"] = optimizer.param_groups[0]["lr"]

            liveloss.update(logs)
            liveloss.send()

            
def one_epoch_test(test_dataloader, model, loss):
    """Evaluate model performance on test dataset.
    
    This function performs comprehensive evaluation of the trained model on the
    test dataset, computing loss, accuracy, and collecting predictions for
    further analysis. It provides detailed performance metrics and returns
    predictions for confusion matrix generation.
    
    Args:
        test_dataloader (torch.utils.data.DataLoader): DataLoader containing
            test data batches.
        model (torch.nn.Module): Trained neural network model to evaluate.
        loss (torch.nn.Module): Loss function for computing test loss.
        
    Returns:
        tuple: A tuple containing:
            - test_loss (float): Average test loss across all batches
            - preds (list): List of predicted class indices
            - actuals (list): List of actual class indices
            
    Note:
        - Disables gradient computation for efficiency
        - Sets model to evaluation mode
        - Automatically moves model and data to best available device (MPS/CUDA/CPU)
        - Computes accuracy as percentage of correct predictions
        - Prints test loss and accuracy to console
        - Returns predictions and ground truth for confusion matrix analysis
        
    Example:
        >>> test_loss, predictions, actuals = one_epoch_test(test_loader, model, criterion)
        >>> print(f"Test Loss: {test_loss:.6f}")
        >>> print(f"Test Accuracy: {accuracy:.2f}%")
    """
    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    # Get the best available device
    device = get_device()

    # we do not need the gradients
    with torch.no_grad():

        # set the model to evaluation mode
        model.eval()  # -

        # move the model to the selected device
        model = model.to(device)

        # Loop over test dataset
        # We also accumulate predictions and targets so we can return them
        preds = []
        actuals = []
        
        for batch_idx, (data, target) in tqdm(
                enumerate(test_dataloader),
                desc='Testing',
                total=len(test_dataloader),
                leave=True,
                ncols=80
        ):
            # move data to the selected device
            data, target = data.to(device), target.to(device)

            # 1. forward pass: compute predicted outputs by passing inputs to the model
            logits = model(data)  # =
            # 2. calculate the loss
            loss_value = loss(logits, target).detach()  # =

            # update average test loss
            test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss_value.data.item() - test_loss))

            # convert logits to predicted class
            # NOTE: the predicted class is the index of the max of the logits
            pred = logits.data.max(1, keepdim=True)[1]  # =

            # compare predictions to true label
            correct += torch.sum(torch.squeeze(pred.eq(target.data.view_as(pred))).cpu())
            total += data.size(0)
            
            preds.extend(pred.data.cpu().numpy().squeeze())
            actuals.extend(target.data.view_as(pred).cpu().numpy().squeeze())

    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))

    return test_loss, preds, actuals


def plot_confusion_matrix(pred, truth, classes):
    """Generate and display confusion matrix for classification results.
    
    This function creates a visual confusion matrix using seaborn heatmap
    to analyze model performance across different classes. It helps identify
    which classes are commonly confused with each other.
    
    Args:
        pred (list or array-like): Predicted class labels from model.
        truth (list or array-like): Ground truth class labels.
        classes (list): List of class names for labeling the matrix axes.
        
    Returns:
        pandas.DataFrame: Confusion matrix as a DataFrame with class names
            as row and column labels.
            
    Note:
        - Uses pandas crosstab for confusion matrix computation
        - Creates heatmap with annotations showing count values
        - Removes color bar for cleaner visualization
        - Sets appropriate axis labels for interpretation
        - Requires matplotlib and seaborn for visualization
        
    Example:
        >>> classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
        >>> cm = plot_confusion_matrix(predictions, ground_truth, classes)
        >>> plt.show()
        >>> print(f"Confusion matrix shape: {cm.shape}")
    """

    gt = pd.Series(truth, name='Ground Truth')
    predicted = pd.Series(pred, name='Predicted')

    confusion_matrix = pd.crosstab(gt, predicted)
    confusion_matrix.index = classes
    confusion_matrix.columns = classes
    
    fig, sub = plt.subplots()
    with sns.plotting_context("notebook"):

        ax = sns.heatmap(
            confusion_matrix, 
            annot=True, 
            fmt='d',
            ax=sub, 
            linewidths=0.5, 
            linecolor='lightgray', 
            cbar=False
        )
        ax.set_xlabel("truth")
        ax.set_ylabel("pred")

    return confusion_matrix
