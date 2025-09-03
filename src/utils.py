import torch

def get_device():
    """
    Determines the most appropriate computation device (MPS, CUDA, or CPU) and returns it.

    The function prioritizes devices in the following order:
    1. MPS (for Apple Silicon GPUs)
    2. CUDA (for NVIDIA GPUs)
    3. CPU (as a fallback)

    Returns:
        torch.device: The selected computation device.
    """
    # Check for MPS availability first (Apple Silicon)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    # Then check for CUDA
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    return device
