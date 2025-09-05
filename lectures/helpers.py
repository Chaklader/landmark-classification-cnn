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
import math
import torch
import torch.utils.data
from pathlib import Path
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import multiprocessing



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
    
def get_train_val_data_loaders(batch_size, valid_size, transforms, num_workers):
    """
    Splits the CIFAR10 training dataset into training and validation sets.
    
    Args:
        batch_size (int): Batch size for data loaders
        valid_size (float): Fraction of data to reserve for validation
        transforms: Image transformations to apply
        num_workers (int): Number of subprocesses to use for data loading
    
    Returns:
        tuple: (train_loader, valid_loader)
    """

    # Get the CIFAR10 training dataset from torchvision.datasets and set the transforms
    # We will split this further into train and validation in this function
    train_data = datasets.CIFAR10("data", train=True, download=True, transform=transforms)

    # Compute how many items we will reserve for the validation set
    n_tot = len(train_data)
    split = int(np.floor(valid_size * n_tot))

    # compute the indices for the training set and for the validation set
    shuffled_indices = torch.randperm(n_tot)
    train_idx, valid_idx = shuffled_indices[split:], shuffled_indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # prepare data loaders (combine dataset and sampler)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers
    )
    valid_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers
    )

    return train_loader, valid_loader


def get_test_data_loader(batch_size, transforms, num_workers):
    """
    Creates a DataLoader for the CIFAR10 test dataset.
    
    Args:
        batch_size (int): Batch size for the DataLoader
        transforms: Image transformations to apply
        num_workers (int): Number of subprocesses to use for data loading
    
    Returns:
        DataLoader: Test DataLoader
    """
    # We use the entire test dataset in the test dataloader
    test_data = datasets.CIFAR10("data", train=False, download=True, transform=transforms)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, num_workers=num_workers
    )

    return test_loader


def train_one_epoch(train_dataloader, model, optimizer, loss):
    """
    Performs one epoch of training
    
    Args:
        train_dataloader (DataLoader): DataLoader for training data
        model: PyTorch model to train
        optimizer: Optimizer to use for training
        loss: Loss function to use
    
    Returns:
        float: Average training loss for the epoch
    """

    # Move model to GPU if available
    if torch.cuda.is_available():
        model.cuda()  # -

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
        # move data to GPU if available
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

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
    """
    Validate at the end of one epoch
    
    Args:
        valid_dataloader (DataLoader): DataLoader for validation data
        model: PyTorch model to validate
        loss: Loss function to use
    
    Returns:
        float: Average validation loss for the epoch
        """

    # During validation we don't need to accumulate gradients
    with torch.no_grad():

        # set the model to evaluation mode
        # (so all layers that behave differently between training and evaluation,
        # like batchnorm and dropout, will select their evaluation behavior)
        model.eval()  # -

        # If the GPU is available, move the model to the GPU
        if torch.cuda.is_available():
            model.cuda()

        # Loop over the validation dataset and accumulate the loss
        valid_loss = 0.0
        for batch_idx, (data, target) in tqdm(
            enumerate(valid_dataloader),
            desc="Validating",
            total=len(valid_dataloader),
            leave=True,
            ncols=80,
        ):
            # move data to GPU if available
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

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
    """
    Trains the model for a specified number of epochs and saves the best model.
    
    Args:
        data_loaders (dict): Dictionary containing 'train' and 'valid' DataLoaders
        model: PyTorch model to train
        optimizer: Optimizer to use for training
        loss: Loss function to use
        n_epochs (int): Number of epochs to train for
        save_path (str): Path to save the best model weights
        interactive_tracking (bool): Whether to use interactive tracking
    
    Returns:
        None
    """
    # initialize tracker for minimum validation loss
    if interactive_tracking:
        liveloss = PlotLosses()
    else:
        liveloss = None

    # Loop over the epochs and keep track of the minimum of the validation loss
    valid_loss_min = None
    logs = {}

    for epoch in range(1, n_epochs + 1):

        train_loss = train_one_epoch(
            data_loaders["train"], model, optimizer, loss
        )

        valid_loss = valid_one_epoch(data_loaders["valid"], model, loss)

        # print training/validation statistics
        print(
            "Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}".format(
                epoch, train_loss, valid_loss
            )
        )

        # If the validation loss decreases by more than 1%, save the model
        if valid_loss_min is None or (
                (valid_loss_min - valid_loss) / valid_loss_min > 0.01
        ):
            print(f"New minimum validation loss: {valid_loss:.6f}. Saving model ...")

            # Save the weights to save_path
            torch.save(model.state_dict(), save_path)  # -

            valid_loss_min = valid_loss

        # Log the losses and the current learning rate
        if interactive_tracking:
            logs["loss"] = train_loss
            logs["val_loss"] = valid_loss

            liveloss.update(logs)
            liveloss.send()

            
def one_epoch_test(test_dataloader, model, loss):
    """
    Performs one epoch of testing
    
    Args:
        test_dataloader (DataLoader): DataLoader for test data
        model: PyTorch model to test
        loss: Loss function to use
    
    Returns:
        tuple: (test_loss, preds, actuals)
    """
    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    # we do not need the gradients
    with torch.no_grad():

        # set the model to evaluation mode
        model.eval()  # -

        # if the GPU is available, move the model to the GPU
        if torch.cuda.is_available():
            model = model.cuda()

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
            # move data to GPU if available
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

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

    gt = pd.Series(truth, name='Ground Truth')
    predicted = pd.Series(pred, name='Predicted')

    confusion_matrix = pd.crosstab(gt, predicted)
    confusion_matrix.index = classes
    confusion_matrix.columns = classes
    
    _, sub = plt.subplots()
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



def anomaly_detection_display(df, n_samples=10):
    """
    Display anomaly detection results with loss distribution and sample images.
    
    This function creates a comprehensive visualization for anomaly detection results
    from an autoencoder, showing:
    1. Distribution of reconstruction losses (histogram)
    2. Most difficult to reconstruct images (anomalies - high loss)
    3. Most typical images (normal - low loss)
    
    Args:
        df (pandas.DataFrame): DataFrame containing columns:
            - 'loss': reconstruction loss values
            - 'input': input images (as tensors or arrays)
            - 'reconstruction': reconstructed images (as tensors or arrays)
        n_samples (int): Number of sample images to display for each category
    
    Returns:
        None: Displays the plots
    """
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: Distribution of reconstruction losses
    ax1 = plt.subplot(3, 1, 1)
    plt.hist(df['loss'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Score (loss)')
    plt.ylabel('Count per bin')
    plt.title('Distribution of score (loss)')
    plt.grid(True, alpha=0.3)
    
    # Sort dataframe by loss for anomaly detection
    df_sorted = df.sort_values('loss', ascending=False)
    
    # Get the most anomalous (highest loss) samples
    anomalies = df_sorted.head(n_samples)
    
    # Get the most normal (lowest loss) samples  
    normals = df_sorted.tail(n_samples)
    
    # Plot 2: Most difficult to reconstruct (anomalies)
    ax2 = plt.subplot(3, 1, 2)
    plt.title('Most difficult to reconstruct')
    plt.axis('off')
    
    # Create a grid for anomaly samples
    for i in range(min(n_samples, len(anomalies))):
        # Input image (top row)
        plt.subplot(3, n_samples * 2, n_samples * 2 + i + 1)
        img_input = anomalies.iloc[i]['input']
        if hasattr(img_input, 'cpu'):  # Handle torch tensors
            img_input = img_input.cpu().detach().numpy()
        if len(img_input.shape) == 3 and img_input.shape[0] in [1, 3]:  # CHW format
            img_input = np.transpose(img_input, (1, 2, 0))
        if img_input.shape[-1] == 1:  # Remove single channel dimension
            img_input = img_input.squeeze(-1)
        
        plt.imshow(img_input, cmap='gray' if len(img_input.shape) == 2 else None)
        plt.axis('off')
        if i == 0:
            plt.ylabel('Input', rotation=0, labelpad=20)
        
        # Reconstruction (bottom row)
        plt.subplot(3, n_samples * 2, n_samples * 3 + i + 1)
        img_recon = anomalies.iloc[i]['reconstruction']
        if hasattr(img_recon, 'cpu'):  # Handle torch tensors
            img_recon = img_recon.cpu().detach().numpy()
        if len(img_recon.shape) == 3 and img_recon.shape[0] in [1, 3]:  # CHW format
            img_recon = np.transpose(img_recon, (1, 2, 0))
        if img_recon.shape[-1] == 1:  # Remove single channel dimension
            img_recon = img_recon.squeeze(-1)
            
        plt.imshow(img_recon, cmap='gray' if len(img_recon.shape) == 2 else None)
        plt.axis('off')
        if i == 0:
            plt.ylabel('Reconst', rotation=0, labelpad=20)
    
    # Add some spacing
    plt.tight_layout()
    
    # Create a new figure for normal samples
    fig2 = plt.figure(figsize=(15, 4))
    plt.suptitle('Sample of in-distribution numbers', fontsize=14)
    
    # Plot normal samples
    for i in range(min(n_samples, len(normals))):
        # Input image (top row)
        plt.subplot(2, n_samples, i + 1)
        img_input = normals.iloc[i]['input']
        if hasattr(img_input, 'cpu'):  # Handle torch tensors
            img_input = img_input.cpu().detach().numpy()
        if len(img_input.shape) == 3 and img_input.shape[0] in [1, 3]:  # CHW format
            img_input = np.transpose(img_input, (1, 2, 0))
        if img_input.shape[-1] == 1:  # Remove single channel dimension
            img_input = img_input.squeeze(-1)
        
        plt.imshow(img_input, cmap='gray' if len(img_input.shape) == 2 else None)
        plt.axis('off')
        if i == 0:
            plt.ylabel('Input', rotation=0, labelpad=20)
        
        # Reconstruction (bottom row)
        plt.subplot(2, n_samples, n_samples + i + 1)
        img_recon = normals.iloc[i]['reconstruction']
        if hasattr(img_recon, 'cpu'):  # Handle torch tensors
            img_recon = img_recon.cpu().detach().numpy()
        if len(img_recon.shape) == 3 and img_recon.shape[0] in [1, 3]:  # CHW format
            img_recon = np.transpose(img_recon, (1, 2, 0))
        if img_recon.shape[-1] == 1:  # Remove single channel dimension
            img_recon = img_recon.squeeze(-1)
            
        plt.imshow(img_recon, cmap='gray' if len(img_recon.shape) == 2 else None)
        plt.axis('off')
        if i == 0:
            plt.ylabel('Reconst', rotation=0, labelpad=20)
    
    plt.tight_layout()
    plt.show()


def show_feature_maps(original_image, feature_maps, filters=None):
    """
    Visualize convolutional feature maps from CNN layers.
    
    This function displays the output feature maps generated by convolutional
    layers, showing how different filters respond to input images. Useful for
    understanding what features the CNN is detecting at different layers.
    
    Args:
        original_image (torch.Tensor or numpy.ndarray): Original input image
            to display alongside feature maps for comparison
        feature_maps (torch.Tensor or numpy.ndarray): Feature maps with shape
            (batch_size, num_channels, height, width) or (num_channels, height, width)
        filters (torch.Tensor or numpy.ndarray, optional): Convolutional filters
            (not used in current implementation but kept for compatibility)
    
    Returns:
        None: Displays the feature map visualization plot
    """
    # Convert to numpy if it's a torch tensor
    if hasattr(feature_maps, 'cpu'):
        feature_maps = feature_maps.cpu().detach().numpy()
    
    # Handle batch dimension - take first sample if batch exists
    if len(feature_maps.shape) == 4:  # (batch, channels, height, width)
        feature_maps = feature_maps[0]  # Take first sample
    
    num_maps = feature_maps.shape[0]  # Number of feature maps
    
    # Calculate grid dimensions
    cols = min(4, num_maps)  # Max 4 columns
    rows = (num_maps + cols - 1) // cols  # Ceiling division
    
    # Add extra row if original image is provided
    if original_image is not None:
        rows += 1
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    
    # Handle single row case
    if rows == 1:
        axes = axes.reshape(1, -1) if num_maps > 1 else [[axes]]
    elif len(axes.shape) == 1:
        axes = axes.reshape(-1, 1)
    
    current_row = 0
    
    # Show original image if provided
    if original_image is not None:
        if hasattr(original_image, 'cpu'):
            original_image = original_image.cpu().detach().numpy()
        
        # Handle different image formats
        if len(original_image.shape) == 4:  # Batch dimension
            original_image = original_image[0]
        
        if len(original_image.shape) == 3:  # (C, H, W) -> (H, W, C)
            if original_image.shape[0] in [1, 3]:  # Channel first
                original_image = np.transpose(original_image, (1, 2, 0))
        
        # Display original image
        axes[current_row][0].imshow(original_image.squeeze(), cmap='gray' if len(original_image.shape) == 2 else None)
        axes[current_row][0].set_title('Original Image')
        axes[current_row][0].axis('off')
        
        # Hide other subplots in the first row
        for col in range(1, cols):
            axes[current_row][col].axis('off')
        
        current_row += 1
    
    # Display feature maps
    for i in range(num_maps):
        row = current_row + (i // cols)
        col = i % cols
        
        # Plot the feature map
        im = axes[row][col].imshow(feature_maps[i], cmap='viridis', interpolation='nearest')
        axes[row][col].set_title(f'Feature Map {i+1}')
        axes[row][col].axis('off')
        
        # Add colorbar for reference
        plt.colorbar(im, ax=axes[row][col], fraction=0.046, pad=0.04)
    
    # Hide empty subplots
    total_plots = num_maps
    if original_image is not None:
        total_plots += cols  # Account for original image row
    
    for i in range(total_plots, rows * cols):
        row = i // cols
        col = i % cols
        if row < len(axes) and col < len(axes[0]):
            axes[row][col].axis('off')
    
    plt.tight_layout()
    plt.show()


def visualize_filters(filters):
    """
    Visualize a set of convolutional filters as grayscale images.
    
    This function displays multiple filters in a grid layout, showing how each
    filter looks as an image. This is useful for understanding what patterns
    different convolutional filters are designed to detect.
    
    Args:
        filters (numpy.ndarray or torch.Tensor): Array of filters with shape
            (num_filters, height, width) or (num_filters, channels, height, width)
    
    Returns:
        None: Displays the filter visualization plot
    """
    # Convert to numpy if it's a torch tensor
    if hasattr(filters, 'cpu'):
        filters = filters.cpu().detach().numpy()
    
    # Handle different filter shapes
    if len(filters.shape) == 4:  # (num_filters, channels, height, width)
        # Take the first channel if multi-channel filters
        filters = filters[:, 0, :, :]
    elif len(filters.shape) == 3:  # (num_filters, height, width)
        pass  # Already in correct format
    else:
        # Single filter case
        filters = filters.reshape(1, filters.shape[-2], filters.shape[-1])
    
    num_filters = filters.shape[0]
    
    # Calculate grid dimensions
    cols = min(4, num_filters)  # Max 4 columns
    rows = (num_filters + cols - 1) // cols  # Ceiling division
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    
    # Handle single row case
    if rows == 1:
        axes = axes.reshape(1, -1) if num_filters > 1 else [[axes]]
    
    for i in range(num_filters):
        row = i // cols
        col = i % cols
        
        # Plot the filter
        im = axes[row][col].imshow(filters[i], cmap='gray', interpolation='nearest')
        axes[row][col].set_title(f'Filter {i+1}')
        axes[row][col].axis('off')
        
        # Add colorbar for reference
        plt.colorbar(im, ax=axes[row][col], fraction=0.046, pad=0.04)
    
    # Hide empty subplots
    for i in range(num_filters, rows * cols):
        row = i // cols
        col = i % cols
        axes[row][col].axis('off')
    
    plt.tight_layout()
    plt.show()


