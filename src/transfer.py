import torch
import torchvision
import torchvision.models as models
import torch.nn as nn


def get_model_transfer_learning(model_name="resnet18", n_classes=50):
    """
    Create a transfer learning model using a pre-trained torchvision model.
    
    This function loads a pre-trained model from torchvision.models, freezes all
    its parameters to prevent training, and replaces the final fully connected
    layer with a new one that outputs the specified number of classes for the
    target classification task.
    
    Args:
        model_name (str, optional): Name of the pre-trained model architecture 
            to use. Must be a valid model name from torchvision.models.
            Defaults to "resnet18".
        n_classes (int, optional): Number of output classes for the final 
            classification layer. Defaults to 50.
    
    Returns:
        torch.nn.Module: A modified pre-trained model with:
            - All original parameters frozen (requires_grad=False)
            - Final fully connected layer replaced with new linear layer
            - Output dimension matching n_classes
    
    Raises:
        ValueError: If model_name is not a valid torchvision model architecture.
            The error message includes a link to the official torchvision models
            documentation for the current version.
    
    Example:
        >>> # Create a ResNet-18 model for 10-class classification
        >>> model = get_model_transfer_learning("resnet18", n_classes=10)
        >>> 
        >>> # Create a VGG-16 model for binary classification
        >>> model = get_model_transfer_learning("vgg16", n_classes=2)
        >>> 
        >>> # Use default ResNet-18 for 50-class classification
        >>> model = get_model_transfer_learning()
    
    Note:
        - The function assumes the model has a 'fc' attribute for the final layer
        - This works for ResNet, VGG, and similar architectures
        - For models with different final layer names (e.g., 'classifier'), 
          this function would need modification
        - All pre-trained weights are preserved and frozen for feature extraction
    """
    
    # Get the requested architecture
    if hasattr(models, model_name):
        model_transfer = getattr(models, model_name)(pretrained=True)
    else:
        torchvision_major_minor = ".".join(torchvision.__version__.split(".")[:2])
        raise ValueError(f"Model {model_name} is not known. List of available models: "
                         f"https://pytorch.org/vision/{torchvision_major_minor}/models.html")

    # Freeze all parameters in the model
    for param in model_transfer.parameters():
        param.requires_grad = False

    # Add the linear layer at the end with the appropriate number of classes
    # 1. get numbers of features extracted by the backbone
    num_ftrs = model_transfer.fc.in_features

    # 2. Create a new linear layer with the appropriate number of inputs and outputs
    model_transfer.fc = nn.Linear(num_ftrs, n_classes)

    return model_transfer


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_get_model_transfer_learning(data_loaders):

    model = get_model_transfer_learning(n_classes=23)

    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
