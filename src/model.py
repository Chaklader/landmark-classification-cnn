import torch
import torch.nn as nn


class MyModel(nn.Module):
    """
    A custom Convolutional Neural Network (CNN) for image classification.

    The architecture is composed of two main parts:
    1. A feature extractor (`self.features`): This consists of four sequential
       convolutional blocks. Each block contains a Conv2d layer, BatchNorm2d for
       stabilizing learning, a ReLU activation function, MaxPool2d for down-sampling,
       and Dropout for regularization. The depth of the feature maps increases
       through the blocks (64 -> 128 -> 256 -> 512) to capture increasingly
       complex patterns.

    2. A classifier (`self.classifier`): This part flattens the output from the
       feature extractor and passes it through a series of fully connected (Linear)
       layers. It reduces the feature dimensions down to the final number of classes.
       Dropout is also applied here to prevent overfitting.

    Args:
        num_classes (int): The number of output classes for the final classification layer.
        dropout (float): The dropout probability to be used in the dropout layers.
    """
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:
        super().__init__()

        """
        Define a CNN architecture. Remember to use the variable num_classes
        to size appropriately the output of your classifier, and if you use
        the Dropout layer, use the variable "dropout" to indicate how much
        to use (like nn.Dropout(p=dropout))

        Feature extractor
        """
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=dropout / 2),  # Less dropout in early layers

            # Second conv block
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=dropout / 2),

            # Third conv block
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=dropout / 2),

            # Fourth conv block
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=dropout)
        )

        """
        The classifier part of the model.

        It takes the flattened output of the feature extractor and passes it
        through a series of fully connected layers to produce the final
        classification scores.

        - Flatten: Converts the 2D feature maps into a 1D vector.
        - Linear (512 * 14 * 14 -> 1024): A fully connected layer that maps
          the flattened features to an intermediate representation. The input size
          is derived from the feature extractor's output:
          - 512: The number of output channels from the last convolutional layer.
          - 14x14: The spatial dimensions (height x width) of the feature map. This
            is calculated from the initial image size of 224x224 being passed
            through four `MaxPool2d` layers. Each layer has a `kernel_size=2` and
            `stride=2`, which halves the dimensions at each step:
            224 -> 112 -> 56 -> 28 -> 14.
        - ReLU: Applies non-linearity.
        - Dropout: Provides regularization to prevent overfitting.
        - Linear (1024 -> num_classes): The final output layer that produces
          scores for each class.

        
        """
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 14 * 14, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # process the input tensor through the feature extractor,
        # the pooling and the final linear layers (if appropriate
        # for the architecture chosen)
        x = self.features(x)
        x = self.classifier(x)
        return x


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):
    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
