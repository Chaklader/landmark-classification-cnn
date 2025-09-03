import tempfile

import torch
import numpy as np
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot
from tqdm import tqdm
from src.helpers import after_subplot
from src.utils import get_device


def train_one_epoch(train_dataloader, model, optimizer, loss):
    """
    Performs a single training epoch on the provided dataset.

    This function iterates through the `train_dataloader`, processing one batch at a
    time. For each batch, it performs the standard training steps: forward pass,
    loss calculation, backward pass (backpropagation), and optimizer step (weight update).
    It also calculates and returns the average training loss for the epoch.

    The model is set to training mode (`model.train()`), which enables features
    like Dropout that are active only during training.

    Args:
        train_dataloader (DataLoader): The data loader for the training set.
        model (nn.Module): The neural network model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer for updating model weights.
        loss (nn.Module): The loss function used for training.

    Returns:
        float: The average training loss over the epoch.
    """
    device = get_device()

    # Move model to the appropriate device and set to training mode
    model.to(device)
    model.train()

    train_loss = 0.0

    for batch_idx, (data, target) in enumerate(tqdm(
            train_dataloader,
            desc="Training",
            total=len(train_dataloader),
            leave=True,
            ncols=80,
    )):
        # Move data and target tensors to the selected device
        data, target = data.to(device), target.to(device)

        # 1. Clear the gradients of all optimized tensors.
        # PyTorch accumulates gradients, so they need to be cleared before each pass.
        optimizer.zero_grad()

        # 2. Perform a forward pass: compute predicted outputs by passing inputs to the model.
        output = model(data)

        # 3. Calculate the batch loss.
        # This measures how far the model's predictions are from the true labels.
        loss_value = loss(output, target)

        # 4. Perform a backward pass: compute gradient of the loss with respect to model parameters.
        # These gradients are used to update the weights in the next step.
        loss_value.backward()

        # 5. Perform a single optimization step: update the model's weights.
        # The optimizer uses the computed gradients to adjust the weights and reduce the loss.
        optimizer.step()

        # update average training loss
        train_loss = train_loss + (
                (1 / (batch_idx + 1)) * (loss_value.data.item() - train_loss)
        )

    return train_loss


def valid_one_epoch(valid_dataloader, model, loss):
    """
    Performs a single validation epoch on the provided dataset.

    This function iterates through the `valid_dataloader` to evaluate the model's
    performance on the validation set. For each batch, it computes the model's
    output and the corresponding loss. It does not perform backpropagation or
    update any model weights.

    The model is set to evaluation mode (`model.eval()`), which disables features
    like Dropout that are only active during training.

    Args:
        valid_dataloader (DataLoader): The data loader for the validation set.
        model (nn.Module): The neural network model to be validated.
        loss (nn.Module): The loss function used for validation.

    Returns:
        float: The average validation loss over the epoch.
    """
    device = get_device()

    # Move model to the appropriate device and set to evaluation mode
    model.to(device)
    model.eval()

    valid_loss = 0.0

    # Iterate over the validation data
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(
                valid_dataloader,
                desc="Validating",
                total=len(valid_dataloader),
                leave=True,
                ncols=80
        )):
            # Move data and target tensors to the selected device
            data, target = data.to(device), target.to(device)

            # 1. Forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)

            # 2. Calculate the batch loss
            loss_value = loss(output, target)

            # Update the running average of the validation loss.
            # This formula provides a numerically stable way to compute the mean loss
            # over batches, preventing large fluctuations early in the epoch.
            valid_loss = valid_loss + (
                    (1 / (batch_idx + 1)) * (loss_value.data.item() - valid_loss)
            )

    return valid_loss


def optimize(data_loaders, model, optimizer, loss, n_epochs, save_path, interactive_tracking=False):
    """
    Orchestrates the training and validation of a model over multiple epochs.

    This function manages the entire training lifecycle, including:
    - Setting the device (CPU, CUDA, MPS).
    - Running training and validation for each epoch.
    - Calculating validation accuracy.
    - Using a learning rate scheduler to adjust the learning rate based on validation loss.
    - Saving the best model checkpoint based on performance improvement.
    - Optionally plotting losses and accuracy live.

    Args:
        data_loaders (dict): A dictionary with 'train' and 'valid' DataLoaders.
        model (nn.Module): The PyTorch model to train.
        optimizer (torch.optim.Optimizer): The optimizer for weight updates.
        loss (nn.Module): The loss function.
        n_epochs (int): The number of epochs to train for.
        save_path (str): Path to save the best model checkpoint.
        interactive_tracking (bool): If True, enables live plotting of metrics.

    Returns:
        nn.Module: The trained model.
    """
    device = get_device()
    model.to(device)

    # Learning rate scheduler: reduces LR when validation loss plateaus.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=10
    )

    # Live plotting setup
    liveloss = PlotLosses(outputs=[MatplotlibPlot(after_subplot=after_subplot)]) if interactive_tracking else None

    # Trackers for best performance
    valid_loss_min = np.Inf
    best_accuracy = 0.0

    print("\nStarting Training Loop...")
    print("=" * 80)

    for epoch in range(1, n_epochs + 1):
        logs = {}
        print(f"\n[Epoch {epoch}/{n_epochs}] Starting training phase...")
        train_loss = train_one_epoch(data_loaders["train"], model, optimizer, loss)

        print(f"[Epoch {epoch}/{n_epochs}] Starting validation phase...")
        valid_loss = valid_one_epoch(data_loaders["valid"], model, loss)

        # Calculate validation accuracy
        correct, total = 0, 0
        model.eval()
        with torch.no_grad():
            for data, target in data_loaders["valid"]:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        accuracy = 100 * correct / total

        print(f"\n[Epoch {epoch}/{n_epochs}] Results:")
        print(f"[Epoch {epoch}/{n_epochs}] Training Loss:       {train_loss:.6f}")
        print(f"[Epoch {epoch}/{n_epochs}] Validation Loss:     {valid_loss:.6f}")
        print(f"[Epoch {epoch}/{n_epochs}] Validation Accuracy: {accuracy:.2f}%")
        print("-" * 80)

        # Check for improvement and save the best model
        if valid_loss < valid_loss_min or accuracy > best_accuracy:
            print(f"\n[Epoch {epoch}/{n_epochs}] Validation improved! Saving model...")
            print(f"    Previous best loss: {valid_loss_min:.6f} | New loss: {valid_loss:.6f}")
            print(f"    Previous best accuracy: {best_accuracy:.2f}% | New accuracy: {accuracy:.2f}%")

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'accuracy': accuracy
            }, save_path)

            valid_loss_min = min(valid_loss, valid_loss_min)
            best_accuracy = max(accuracy, best_accuracy)

        # Update learning rate
        scheduler.step(valid_loss)

        # Update live plot
        if interactive_tracking:
            logs["loss"] = train_loss
            logs["val_loss"] = valid_loss
            logs["val_acc"] = accuracy
            logs["lr"] = optimizer.param_groups[0]["lr"]
            liveloss.update(logs)
            liveloss.send()

        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n[Epoch {epoch}/{n_epochs}] Current Learning Rate: {current_lr}")
        if current_lr < 1e-6:
            print("\nLearning rate is too small. Stopping training early.")
            break
        print("\n" + "=" * 80)

    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_accuracy:.2f}%")
    print(f"Best validation loss: {valid_loss_min:.6f}")

    return model


def one_epoch_test(test_dataloader, model, loss):
    """
    Evaluates the model's performance on the test dataset for a single epoch.

    This function calculates the test loss and accuracy. The model is set to
    evaluation mode, and no gradients are computed.

    Args:
        test_dataloader (DataLoader): The DataLoader for the test data.
        model (nn.Module): The trained model to be tested.
        loss (nn.Module): The loss function used for evaluation.

    Returns:
        float: The average test loss over the epoch.
    """
    device = get_device()
    model.to(device)
    model.eval()

    # Monitor test loss and accuracy
    test_loss = 0.0
    correct = 0.0
    total = 0.0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(
                test_dataloader,
                desc='Testing',
                total=len(test_dataloader),
                leave=True,
                ncols=80
        )):
            # Move data and target tensors to the selected device
            data, target = data.to(device), target.to(device)

            # 1. Forward pass: compute predicted outputs
            logits = model(data)

            # 2. Calculate the loss
            loss_value = loss(logits, target)

            # Update average test loss using a running average
            test_loss += (1 / (batch_idx + 1)) * (loss_value.item() - test_loss)

            # Convert logits to predicted class
            pred = torch.argmax(logits, dim=1)

            # Compare predictions to true labels
            correct += torch.sum(torch.squeeze(pred.eq(target.data.view_as(pred))).cpu()).item()
            total += data.size(0)

    accuracy = 100. * correct / total
    print(f'Test Loss: {test_loss:.6f}\n')
    print(f'Test Accuracy: {accuracy:.2f}% ({int(correct)}/{int(total)})')

    return test_loss


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=50, limit=200, valid_size=0.5, num_workers=0)


@pytest.fixture(scope="session")
def optim_objects():
    from src.optimization import get_optimizer, get_loss
    from src.model import MyModel

    model = MyModel(50)

    return model, get_loss(), get_optimizer(model)


def test_train_one_epoch(data_loaders, optim_objects):
    model, loss, optimizer = optim_objects

    for _ in range(2):
        lt = train_one_epoch(data_loaders['train'], model, optimizer, loss)
        assert not np.isnan(lt), "Training loss is nan"


def test_valid_one_epoch(data_loaders, optim_objects):
    model, loss, optimizer = optim_objects

    for _ in range(2):
        lv = valid_one_epoch(data_loaders["valid"], model, loss)
        assert not np.isnan(lv), "Validation loss is nan"


def test_optimize(data_loaders, optim_objects):
    model, loss, optimizer = optim_objects

    with tempfile.TemporaryDirectory() as temp_dir:
        optimize(data_loaders, model, optimizer, loss, 2, f"{temp_dir}/hey.pt")


def test_one_epoch_test(data_loaders, optim_objects):
    model, loss, optimizer = optim_objects

    tv = one_epoch_test(data_loaders["test"], model, loss)
    assert not np.isnan(tv), "Test loss is nan"
