# ---------------------------------------------------------
# Yuxuan Zhang
# Dept. of Electrical and Computer Engineering
# University of Florida
# ---------------------------------------------------------
# Derives environment variables and parameters from various
# sources, and maintains the file structure of the project.
# ---------------------------------------------------------
from pathlib import Path
from os import mkdir, environ
from os.path import exists, dirname, realpath
import torch

DEBUG = "DEBUG" in environ
if DEBUG:
    print("In [DEBUG] mode")


def getDevice(force_cpu=False):
    """Get torch device according to system that it runs on"""
    if force_cpu:
        return 'cpu'
    if torch.cuda.is_available():
        # Nvidia Cuda
        if "CUDA_DEVICE" in environ:
            return "cuda:{}".format(environ["CUDA_DEVICE"])
        elif "CUDA_DEV" in environ:
            return "cuda:{}".format(environ["CUDA_DEV"])
        else:
            return 'cuda'
    if torch.backends.mps.is_available():
        # MacOS metal
        return 'mps'
    # CPU
    return 'cpu'


# PyTorch device
DEVICE = torch.device(getDevice(DEBUG))
# Base path of the project
BASE = Path(dirname(dirname(realpath(__file__))))
# Path constants
DATA_PATH = BASE / "data"
VAR_PATH = BASE / "var"
RUN_PATH = BASE / "run"


def ensure(path) -> Path:
    if not isinstance(path, Path):
        path = Path(path)
    if not exists(path):
        mkdir(path)
    return path


# Create paths if not exist
for d in [DATA_PATH, VAR_PATH, RUN_PATH]:
    ensure(d)


def relative(path: Path, base: Path = BASE):
    try:
        if not isinstance(path, Path):
            path = Path(path)
        return path.relative_to(base)
    except:
        return path

def try_paths(*paths: Path, verify = exists):
    for path in paths:
        if verify(path):
            return path
    return None
