import tempfile

import torch
import numpy as np
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot
from tqdm import tqdm
from src.helpers import after_subplot


def train_one_epoch(train_dataloader, model, optimizer, loss):
    """
    Performs one train_one_epoch epoch
    """
    # Check for MPS availability first (Apple Silicon)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    # Then check for CUDA
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Move model to the appropriate device
    model = model.to(device)
    model.train()

    train_loss = 0.0

    for batch_idx, (data, target) in enumerate(tqdm(
            train_dataloader,
            desc="Training",
            total=len(train_dataloader),
            leave=True,
            ncols=80,
    )):
        # Move data to appropriate device
        data, target = data.to(device), target.to(device)

        # 1. clear the gradients
        optimizer.zero_grad()

        # 2. forward pass
        output = model(data)

        # 3. calculate the loss
        loss_value = loss(output, target)

        # 4. backward pass
        loss_value.backward()

        # 5. perform optimization step
        optimizer.step()

        # update average training loss
        train_loss = train_loss + (
                (1 / (batch_idx + 1)) * (loss_value.data.item() - train_loss)
        )

    return train_loss


def valid_one_epoch(valid_dataloader, model, loss):
    """
    Validate at the end of one epoch
    """
    # Check for MPS availability first (Apple Silicon)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    # Then check for CUDA
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    with torch.no_grad():
        # Set model to evaluation mode
        model.eval()

        # Move model to appropriate device
        model = model.to(device)

        valid_loss = 0.0
        for batch_idx, (data, target) in enumerate(tqdm(
                valid_dataloader,
                desc="Validating",
                total=len(valid_dataloader),
                leave=True,
                ncols=80
        )):
            # Move data to appropriate device
            data, target = data.to(device), target.to(device)

            # 1. forward pass
            output = model(data)

            # 2. calculate the loss
            loss_value = loss(output, target)

            # Calculate average validation loss
            valid_loss = valid_loss + (
                    (1 / (batch_idx + 1)) * (loss_value.data.item() - valid_loss)
            )

    return valid_loss


def optimize(data_loaders, model, optimizer, loss, n_epochs, save_path, interactive_tracking=False):
    # Set up device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"\nStarting Training on device: {device}")
    model = model.to(device)

    if interactive_tracking:
        liveloss = PlotLosses(outputs=[MatplotlibPlot(after_subplot=after_subplot)])
    else:
        liveloss = None

    valid_loss_min = None
    best_accuracy = 0.0
    logs = {}

    """
    This is a learning rate scheduler that automatically reduces the learning rate when the validation loss 
    stops improving. Here's what each parameter does:

        - optimizer: The optimizer whose learning rate will be adjusted (e.g., Adam/SGD)
        - mode='min': Monitors validation loss (will reduce LR when loss stops decreasing)
        - factor=0.1: New LR = old LR × 0.1 (a 10x reduction when triggered)
        - patience=10: Waits 10 epochs without improvement before reducing LR
        - verbose=True: Prints messages when LR changes
    """
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=10,
        verbose=True
    )

    print("\nStarting Training Loop...")
    print("=" * 80)

    for epoch in range(1, n_epochs + 1):
        print(f"\n[Epoch {epoch}/{n_epochs}] Starting training phase...")
        train_loss = train_one_epoch(data_loaders["train"], model, optimizer, loss)

        print(f"[Epoch {epoch}/{n_epochs}] Starting validation phase...")
        valid_loss = valid_one_epoch(data_loaders["valid"], model, loss)

        correct = 0
        total = 0
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

        """
        This condition checks if the model's performance has improved in 3 ways:
            - First epoch (valid_loss_min is None)
            - Validation loss decreased by >1% ((valid_loss_min - valid_loss)/valid_loss_min > 0.01)
            - New accuracy record (accuracy > best_accuracy)

        It triggers model checkpointing when any condition is met.
        """
        if (valid_loss_min is None or
                ((valid_loss_min - valid_loss) / valid_loss_min > 0.01) or
                accuracy > best_accuracy):

            print(f"\n[Epoch {epoch}/{n_epochs}] Validation improved!")
            if valid_loss_min is None:
                print(f"[Epoch {epoch}/{n_epochs}] Previous best loss: None")
            else:
                print(f"[Epoch {epoch}/{n_epochs}] Previous best loss: {valid_loss_min:.6f}")
            print(f"[Epoch {epoch}/{n_epochs}] New loss:          {valid_loss:.6f}")
            print(f"[Epoch {epoch}/{n_epochs}] Previous best accuracy: {best_accuracy:.2f}%")
            print(f"[Epoch {epoch}/{n_epochs}] New accuracy:          {accuracy:.2f}%")
            print(f"[Epoch {epoch}/{n_epochs}] Saving model checkpoint...")

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'accuracy': accuracy
            }, save_path)

            valid_loss_min = valid_loss
            best_accuracy = accuracy

        scheduler.step(valid_loss)

        if interactive_tracking:
            logs["loss"] = train_loss
            logs["val_loss"] = valid_loss
            logs["val_acc"] = accuracy
            logs["lr"] = optimizer.param_groups[0]["lr"]
            liveloss.update(logs)
            liveloss.send()

        print(f"\n[Epoch {epoch}/{n_epochs}] Current Learning Rate: {optimizer.param_groups[0]['lr']}")

        if optimizer.param_groups[0]["lr"] < 1e-6:
            print(f"\n[Epoch {epoch}/{n_epochs}] Learning rate too small. Stopping training.")
            break

        print("\n" + "=" * 80)

    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_accuracy:.2f}%")
    print(f"Best validation loss: {valid_loss_min:.6f}")

    return model


def one_epoch_test(test_dataloader, model, loss):
    """
    Test the model for one epoch.

    Args:
        test_dataloader: DataLoader for test data
        model: the trained model
        loss: loss function

    Returns:
        test_loss: average test loss
    """
    # Check for MPS availability first (Apple Silicon)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    # Then check for CUDA
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    # set the module to evaluation mode
    with torch.no_grad():
        # set the model to evaluation mode
        model.eval()

        # Move model to appropriate device
        model = model.to(device)

        for batch_idx, (data, target) in enumerate(tqdm(
                test_dataloader,
                desc='Testing',
                total=len(test_dataloader),
                leave=True,
                ncols=80
        )):
            # Move data to appropriate device
            data, target = data.to(device), target.to(device)

            # 1. forward pass: compute predicted outputs by passing inputs to the model
            logits = model(data)

            # 2. calculate the loss
            loss_value = loss(logits, target)

            # update average test loss
            test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss_value.data.item() - test_loss))

            # convert logits to predicted class
            pred = torch.argmax(logits, dim=1)

            # compare predictions to true label
            correct += torch.sum(torch.squeeze(pred.eq(target.data.view_as(pred))).cpu())
            total += data.size(0)

    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))

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
