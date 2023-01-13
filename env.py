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