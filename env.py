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
        # Nvidia Cuda
        return 'cuda'
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
RUN_PATH = VAR_PATH / "run"

def ensureDir(path):
    if not exists(path):
        mkdir(path)

# Create paths if not exist
[ensureDir(d) for d in [
    VAR_PATH,
    RUN_PATH
]]
