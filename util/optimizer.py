# ---------------------------------------------------------
# Yuxuan Zhang
# Dept. of Electrical and Computer Engineering
# University of Florida
# ---------------------------------------------------------
# TODO: Add description
# ---------------------------------------------------------
from os.path import isfile
from pathlib import Path

# PIP modules
import torch

# Custom modules
from lib import Context
from util.env import RUN_PATH, relative, ensure


SAVE_DIR = "states"


def optimizer(optim: type[torch.nn.Module]):
    class_name = optim.__name__
    state_dict_name = f"{class_name}.pkl"

    class CustomOptim(optim):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.__class__.__name__ = class_name

        def save(self, ctx: Context, path: Path = None):
            if path is None:
                path = ensure(ctx.path / SAVE_DIR)
            else:
                path = ensure(path)
            # Save model
            optim_path = path / f"{class_name}.pkl"
            optim = self.state_dict()
            torch.save(optim, optim_path)
            ctx.log(f"Optim state of {class_name} saved to {relative(optim_path)}")

        def load(self, ctx: Context, **kwargs):
            # Find matching work dir
            if class_name in kwargs:
                runID, remap = kwargs[class_name]
            elif "optim" in kwargs:
                runID, remap = kwargs["optim"]
            elif "optimizer" in kwargs:
                runID, remap = kwargs["optimizer"]
            elif "" in kwargs:
                runID, remap = kwargs[""]
            else:
                return ctx.log(
                    f"[WARNING] No runID specified for loading {class_name}, skipping..."
                )
            # load state dictionary
            dir = RUN_PATH / runID / SAVE_DIR
            if remap is not None and isfile(dir / f"{remap}.pkl"):
                load_path = dir / f"{remap}.pkl"
            elif isfile(dir / state_dict_name):
                load_path = dir / state_dict_name
            else:
                return ctx.log(
                    f"[WARNING] No state dictionary found for {class_name} at {relative(dir)}, skipping..."
                )
            state_dict = torch.load(load_path)
            self.load_state_dict(state_dict)
            ctx.log(f"Optim state of {class_name} loaded from {relative(load_path)}")

    return CustomOptim
