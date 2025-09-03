import torch
import torch.nn as nn
import torch.optim
import torch.optim


def get_loss():
    """
    Creates and returns a loss function suitable for multi-class classification.

    This function returns an instance of `nn.CrossEntropyLoss`. This loss function
    is standard for classification tasks because it combines a LogSoftmax layer and a
    Negative Log-Likelihood Loss (NLLLoss) in one efficient class. It measures the
    difference between the model's predicted probability distribution and the actual
    class distribution, penalizing incorrect predictions.

    Returns:
        nn.CrossEntropyLoss: An instance of the cross-entropy loss function.
    """

    # loss appropriate for classification
    loss = nn.CrossEntropyLoss()

    return loss


def get_optimizer(
        model: nn.Module,
        optimizer: str = "SGD",
        learning_rate: float = 0.01,
        momentum: float = 0.5,
        weight_decay: float = 0,
):
    """
    Creates and returns a PyTorch optimizer for training the model.

    This function acts as a factory for creating an optimizer instance based on the
    provided specifications. It supports two common optimizers:
    - 'SGD' (Stochastic Gradient Descent): A foundational optimizer that updates model
      weights based on the gradient of the loss. It can include momentum to
      accelerate convergence and weight decay for L2 regularization.
    - 'Adam' (Adaptive Moment Estimation): A more advanced optimizer that adapts the
      learning rate for each parameter individually, often leading to faster convergence.

    Args:
        model (nn.Module): The neural network model whose parameters will be optimized.
        optimizer (str): The name of the optimizer to use ('SGD' or 'Adam').
        learning_rate (float): The step size for updating the model's weights.
        momentum (float): The momentum factor for the SGD optimizer.
        weight_decay (float): The coefficient for L2 regularization (weight decay).

    Returns:
        torch.optim.Optimizer: An instance of the specified optimizer.

    Raises:
        ValueError: If the specified optimizer name is not supported.
    """
    if optimizer.lower() == "sgd":
        # create an instance of the SGD
        # optimizer. Use the input parameters learning_rate, momentum
        # and weight_decay
        opt = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )

    elif optimizer.lower() == "adam":
        # YOUR CODE HERE: create an instance of the Adam
        # optimizer. Use the input parameters learning_rate, momentum
        # and weight_decay
        opt = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            betas=(momentum, 0.999),  # using momentum as first beta
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Optimizer {optimizer} not supported")

    return opt


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def fake_model():
    return nn.Linear(16, 256)


def test_get_loss():
    loss = get_loss()

    assert isinstance(
        loss, nn.CrossEntropyLoss
    ), f"Expected cross entropy loss, found {type(loss)}"


def test_get_optimizer_type(fake_model):
    opt = get_optimizer(fake_model)

    assert isinstance(opt, torch.optim.SGD), f"Expected SGD optimizer, got {type(opt)}"


def test_get_optimizer_is_linked_with_model(fake_model):
    opt = get_optimizer(fake_model)

    assert opt.param_groups[0]["params"][0].shape == torch.Size([256, 16])


def test_get_optimizer_returns_adam(fake_model):
    opt = get_optimizer(fake_model, optimizer="adam")

    assert opt.param_groups[0]["params"][0].shape == torch.Size([256, 16])
    assert isinstance(opt, torch.optim.Adam), f"Expected SGD optimizer, got {type(opt)}"


def test_get_optimizer_sets_learning_rate(fake_model):
    opt = get_optimizer(fake_model, optimizer="adam", learning_rate=0.123)

    assert (
            opt.param_groups[0]["lr"] == 0.123
    ), "get_optimizer is not setting the learning rate appropriately. Check your code."


def test_get_optimizer_sets_momentum(fake_model):
    opt = get_optimizer(fake_model, optimizer="SGD", momentum=0.123)

    assert (
            opt.param_groups[0]["momentum"] == 0.123
    ), "get_optimizer is not setting the momentum appropriately. Check your code."


def test_get_optimizer_sets_weight_decat(fake_model):
    opt = get_optimizer(fake_model, optimizer="SGD", weight_decay=0.123)

    assert (
            opt.param_groups[0]["weight_decay"] == 0.123
    ), "get_optimizer is not setting the weight_decay appropriately. Check your code."
