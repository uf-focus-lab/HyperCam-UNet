import sys
from pathlib import Path
from os import mkdir, environ
from os.path import exists, dirname, realpath
import torch

DEBUG = "DEBUG" in environ
if DEBUG:
    print("In [DEBUG] mode")

def getDevice(force_cpu = False):
    """Get torch device according to system that it runs on"""
    if force_cpu:
        return 'cpu'
    if torch.cuda.is_available():
        dev_number = None
        # Nvidia Cuda
        if "CUDA_DEVICE" in environ:
            dev_number = environ["CUDA_DEVICE"]
        elif "CUDA_DEV" in environ:
            dev_number = environ["CUDA_DEV"]
        else:
            dev_count = torch.cuda.device_count()
            if dev_count == 1:
                return "cuda"
            # List available devices
            for i in range(dev_count):
                print(f"{i:2d} |", torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
            # Prompt for selection
            dev_number = input("Select a cuda device from above: ")
        # Return string descriptor
        return f"cuda:{dev_number}" if dev_number is not None else "cuda"
    if torch.backends.mps.is_available():
        # MacOS metal
        return 'mps'
    # CPU
    return 'cpu'

# PyTorch device
DEVICE = torch.device(getDevice(DEBUG))
# Base path of the project
BASE = Path(dirname(realpath(__file__)))
# Path constants
VAR_PATH = BASE / "var"
DATA_PATH = BASE / "data"
RAW_DATA_PATH = DATA_PATH / "raw"
REF_DATA_PATH = DATA_PATH / "ref"
RUN_PATH = VAR_PATH / "run"

def ensureDir(path) -> Path:
    if not exists(str(path)):
        mkdir(str(path))
    return Path(path)

# Create paths if not exist
[ensureDir(d) for d in [
    VAR_PATH,
    RUN_PATH
]]

flag_sigint = False

def set_sigint(*args):
    global flag_sigint
    if flag_sigint:
        sys.exit(0)
    else:
        flag_sigint = True
        print("SIG_INT flag set. Press CTRL-C again to exit immediately")

def check_sigint():
    global flag_sigint
    if flag_sigint:
        flag_sigint = False
        return True
    return False