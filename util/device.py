# ---------------------------------------------------------
# Yuxuan Zhang
# Dept. of Electrical and Computer Engineering
# University of Florida
# ---------------------------------------------------------
# TODO: Add description
# ---------------------------------------------------------
from os import environ
import torch
from util.env import DEBUG


def getCudaNumber():
    # Nvidia Cuda
    if "CUDA_DEVICE" in environ:
        return environ["CUDA_DEVICE"]
    elif "CUDA_DEV" in environ:
        return environ["CUDA_DEV"]
    else:
        # Print all available devices and ask user to select
        dev_count = torch.cuda.device_count()
        if dev_count == 1:
            return None
        # List available devices
        for i in range(dev_count):
            print(f"[{i}]",
                  torch.cuda.get_device_name(i),
                  torch.cuda.get_device_capability(i)
                  )
        # Prompt for selection
        return input("Select a cuda device from above: ")


def getDeviceID(force_cpu=False):
    """Get torch device according to system that it runs on"""
    if force_cpu:
        return 'cpu'
    if torch.cuda.is_available():
        # Nvidia Cuda
        dev_number = getCudaNumber()
        if dev_number is None:
            return 'cuda'
        else:
            return f"cuda:{dev_number}"
    else:
        assert "CUDA_DEVICE" not in environ and "CUDA_DEV" not in environ, \
            "CUDA is not available, but CUDA_DEVICE or CUDA_DEV is set.\n" \
            "CHECK YOUR ENVIRONMENT VARIABLES OR DRIVER CONFIGURATIONS!"
    if torch.backends.mps.is_available():
        # MacOS metal
        return 'mps'
    # No GPU available, fallback to CPU
    return 'cpu'


# PyTorch device
DEVICE = torch.device(getDeviceID(DEBUG))
