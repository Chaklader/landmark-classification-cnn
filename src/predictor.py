import os

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import datasets
import torchvision.transforms as T
from .helpers import get_data_location


class Predictor(nn.Module):
    """A wrapper class for model inference with built-in preprocessing.
    
    This class wraps a trained model and provides preprocessing transforms
    that are compatible with TorchScript. It applies the necessary image
    transformations and returns softmax probabilities for predictions.
    
    Parameters
    ----------
    model : torch.nn.Module
        The trained model to wrap for inference.
    class_names : list of str
        List of class names corresponding to model outputs.
    mean : torch.Tensor
        Per-channel mean values for normalization.
    std : torch.Tensor
        Per-channel standard deviation values for normalization.
    """

    def __init__(self, model, class_names, mean, std):
        super().__init__()

        self.model = model.eval()
        self.class_names = class_names

        # We use nn.Sequential and not nn.Compose because the former
        # is compatible with torch.script, while the latter isn't
        self.transforms = nn.Sequential(
            T.Resize([256, ]),  # We use single int value inside a list due to torchscript type restrictions
            T.CenterCrop(224),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean.tolist(), std.tolist())
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the predictor.
        
        Applies preprocessing transforms, runs inference through the model,
        and returns softmax probabilities.
        
        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (batch_size, channels, height, width).
            
        Returns
        -------
        torch.Tensor
            Softmax probabilities of shape (batch_size, num_classes).
        """
        with torch.no_grad():
            # 1. apply transforms
            x = self.transforms(x)

            # 2. get the logits
            x = self.model(x)

            # 3. apply softmax
            # HINT: remember to apply softmax across dim=1
            x = F.softmax(x, dim=1)

            return x

def predictor_test(test_dataloader, model_reloaded):
    """Tests the predictor model on the test dataset.
    
    This function evaluates the predictor's performance by loading test images
    directly from the filesystem and comparing predictions with ground truth labels.
    Since the predictor operates on raw images rather than preprocessed tensors,
    it requires a specialized test function.
    
    Parameters
    ----------
    test_dataloader : torch.utils.data.DataLoader
        DataLoader for the test dataset (used only for dataset size reference).
    model_reloaded : Predictor
        The predictor model to test.
        
    Returns
    -------
    tuple of numpy.ndarray
        A tuple containing (ground_truth_labels, predicted_labels).
    """

    folder = get_data_location()
    test_data = datasets.ImageFolder(os.path.join(folder, "test"), transform=T.ToTensor())

    pred = []
    truth = []
    for x in tqdm(test_data, total=len(test_dataloader.dataset), leave=True, ncols=80):
        softmax = model_reloaded(x[0].unsqueeze(dim=0))

        idx = softmax.squeeze().argmax()

        pred.append(int(x[1]))
        truth.append(int(idx))

    pred = np.array(pred)
    truth = np.array(truth)

    print(f"Accuracy: {(pred==truth).sum() / pred.shape[0]}")

    return truth, pred


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    from .model import MyModel
    from .helpers import compute_mean_and_std

    mean, std = compute_mean_and_std()

    model = MyModel(num_classes=3, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    predictor = Predictor(model, class_names=['a', 'b', 'c'], mean=mean, std=std)

    out = predictor(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 3]
    ), f"Expected an output tensor of size (2, 3), got {out.shape}"

    assert torch.isclose(
        out[0].sum(),
        torch.Tensor([1]).squeeze()
    ), "The output of the .forward method should be a softmax vector with sum = 1"
