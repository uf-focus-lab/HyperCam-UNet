# ---------------------------------------------------------
# Yuxuan Zhang
# Dept. of Electrical and Computer Engineering
# University of Florida
# ---------------------------------------------------------
# Prototype of the custom Module class based on the pytorch
# nn.Module
# ---------------------------------------------------------
# Internal Modules
from os.path import isdir, isfile

# PIP Modules
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import cv2

# Local Modules
import cvtb
from lib import Context
from util.env import ensure, Path, relative, try_paths
from util.dataset import DataSet


SAVE_DIR = "states"


class Module(torch.nn.Module):
    def __init__(self, ctx=None, *args, device=None, sample=None, loss=None):
        super().__init__()
        self.device = device
        self.loss = loss

    def save(self, ctx: Context, path: Path = None, suffix=[]):
        """Overload this function for custom loading / saving"""
        name = self.__class__.__name__
        if path is None:
            path = ensure(ctx.path / SAVE_DIR)
        else:
            path = ensure(path)
        # Save model
        model_path = path / (
            f"{name}@{'.'.join(suffix)}.pkl" if len(suffix) else f"{name}.pkl"
        )
        states = {}

        # Go through all the modules and call save if it is an instance of our custom Module
        for child_name, module in self.named_children():
            if isinstance(module, Module):
                module.save(ctx, path, suffix + [child_name])
            else:
                state_dict = module.state_dict()
                for key, item in state_dict.items():
                    states[f"{child_name}.{key}"] = item

        torch.save(states, model_path)
        ctx.log(f"Model state of {name} saved to {relative(model_path)}")

    def update(self, ctx: Context, load_path: Path, full_name: str = None):
        """Update the model with the given load_path"""
        states_to_load = torch.load(load_path)
        states = self.state_dict()
        states.update(states_to_load)
        self.load_state_dict(states)
        # Log a successful load
        if full_name is None:
            full_name = self.__class__.__name__
        ctx.log(f"Model state of {full_name} loaded from {relative(load_path)}")

    def load(self, ctx: Context, suffix=[], **path_list: str):
        """Overload this function for custom loading / saving"""
        # Go through all the modules and call load if it is an instance of our custom Module
        for name, module in self.named_children():
            if isinstance(module, Module):
                module.load(ctx, suffix + [name], **path_list)
        # Get the names of this model
        name = self.__class__.__name__
        full_name = f"{name}@{'.'.join(suffix)}" if len(suffix) else name
        # Try to get runID and remap from the path_list
        if full_name in path_list:
            load_path, remap = path_list[full_name]
        if name in path_list:
            load_path, remap = path_list[name]
        elif "" in path_list:
            load_path, remap = path_list[""]
        else:
            ctx.log(f"[WARNING] No runID specified for loading {name}, skipping...")
            return
        # Get the path to the specified run to be loaded
        dir = load_path / SAVE_DIR
        if not isdir(dir):
            ctx.log(
                f"[WARNING] Load path ({dir}) for {name} does not exist, skipping..."
            )
            return
        # Load model
        if remap is not None:
            # Try exact match first
            full_name = remap
            # Try use remap as suffix
            name = name + "@" + remap
        path = try_paths(dir / f"{full_name}.pkl", dir / f"{name}.pkl")
        if path is not None:
            try:
                self.update(ctx, path, full_name)
            except Exception as e:
                ctx.log("[WARNING] Failed to load", full_name, "from", relative(dir), ':', e)
        else:
            # Try to load from anything that was saved by the module
            for path in dir.glob(f"{self.__class__.__name__}@*.pkl"):
                try:
                    ctx.log(
                        "Attempting to load", relative(path), "for", full_name, "..."
                    )
                    self.update(ctx, path, full_name)
                    return
                except:
                    pass
            ctx.log(
                "[WARNING]",
                relative(dir),
                "does not have files that can be loaded for",
                full_name,
                "skipping...",
            )

    # Virtual Function
    def lossFunction(self, pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        """Calculates loss from given prediction against ground truth"""
        if self.loss is None:
            print(self.loss)
            raise NotImplementedError
        return self.loss(pred, truth)

    # Virtual Function - Optional
    def preview(self, input: torch.Tensor, pred: torch.Tensor, truth: torch.Tensor):
        """
        [ Virtual | Optional ]
        Generate preview images
        """
        return None

    def transform(self, ctx: Context, *data_point: torch.Tensor) -> torch.Tensor:
        """
        [ Virtual | Optional ]
        Transform data point before feeding into the iterate_batch
        """
        data_point = [
            x.to(self.device) if isinstance(x, torch.Tensor) else x for x in data_point
        ]
        return data_point

    # Virtual Function - Optional
    def iterate_epoch(self, ctx: Context, epoch, loader: DataSet | DataLoader):
        """
        [ Virtual | Optional ]
        Iterate one epoch
        """
        # Prefix
        prefix = f"Epoch {epoch:4d}" if ctx.train_mode else "Test ----"
        # Progress bar
        prog = tqdm(
            loader,
            leave=False,
            ncols=80,
            bar_format="{l_bar} |{bar}| {n_fmt}/{total_fmt} {rate_fmt}{postfix}",
        )
        prog.set_description(f"{ctx.id} | {prefix}")
        hist = {}
        for data_point in prog:
            # Transform data point
            data_point = self.transform(ctx, *data_point)
            # Iterate batch
            if ctx.train_mode:
                score = self.iterate_batch(ctx, *data_point)
            else:
                with torch.no_grad():
                    score = self.iterate_batch(ctx, *data_point)
            # Record score of this batch
            for k in score:
                if k in hist:
                    hist[k] += [score[k]]
                else:
                    hist[k] = [score[k]]
            # Check for user signal
            if ctx.signal.triggered:
                break
        # Generate epoch report
        report = [f"{k} {np.average(hist[k]):.6E}" for k in hist]
        ctx.log(prefix, "|", " | ".join(report))
        # Save all intermediate data pushed into context during training.
        for key, mem in ctx.collect_all():
            if all(isinstance(x, np.ndarray) for x in mem):
                # Save as a single numpy array
                res = np.concatenate(mem, axis=0)
                np.save(ctx.path / key, res)
                ctx.log(f"Saved {key}: {res.shape} ({res.dtype})")
            else:
                # Save as plain text
                for s in mem:
                    ctx.log(s, file=f"{key}.txt", visible=False)
                ctx.log(f"Saved {key} as {len(mem)} lines of text")
        # Generate preview
        with torch.no_grad():
            if isinstance(loader, DataSet):
                sample, truth = loader.sample(ctx.sample_size)
            else:
                assert isinstance(loader, DataLoader), loader
                sample, truth = loader.dataset.sample(ctx.sample_size)
            sample, truth = self.transform(ctx, sample, truth)
            prediction = self(sample, train=ctx.train_mode).detach()
            preview = self.preview(sample, prediction, truth)
            if preview is not None:
                cv2.imwrite(str(ctx.path / "preview.png"), cvtb.types.U8(preview))

    def iterate_batch(self, ctx: Context, *data_point: torch.Tensor):
        """
        [ Virtual | Optional ]
        Iterate one batch
        """
        batch, truth = list(data_point)[:2]
        batch = batch.to(self.device)
        # Forward pass
        prediction = self(batch, train=ctx.train_mode)
        # Release batch memory
        del batch
        # Compute truth
        truth = truth.to(self.device)
        # Compute loss
        loss = self.lossFunction(prediction, truth)
        # Check for run mode
        if ctx.train_mode:
            # Clear previously computed gradient
            ctx.optimizer.zero_grad()
            # Backward Propagation
            loss.backward()
            # Optimizing the parameters
            ctx.optimizer.step()
        else:
            prediction = prediction.detach().cpu().numpy()
            name = self.__class__.__name__
            ctx.push(f"{name}.prediction", prediction)
        # Report results
        return {"loss": loss.detach().cpu().numpy()}

    def forward(self, x: torch.Tensor, train=None):
        raise NotImplementedError

    def run(self, ctx: Context, loader: DataLoader | DataSet):
        """
        Run the model in either training mode or testing mode
        """
        # Check if loader is torch DataLoader
        if not isinstance(loader, DataLoader):
            assert isinstance(loader, DataSet), loader
            loader = DataLoader(loader, batch_size=1, shuffle=False)
        # Get number of epochs from runtime configuration
        epochs = ctx.epochs if isinstance(ctx.epochs, int) else 1
        # Main loop
        with ctx.signal:
            for epoch in range(1, epochs + 1):
                self.iterate_epoch(ctx, epoch, loader)
                if ctx.signal.triggered:
                    break

    def __setattr__(self, key, value):
        if hasattr(self, "device") and isinstance(self.device, torch.device):
            if isinstance(value, Module):
                value.device = self.device
            if isinstance(value, torch.nn.Module):
                value = value.to(self.device)
        # Direct only nn.Module to super().__setattr__()
        if isinstance(value, torch.nn.Module):
            super().__setattr__(key, value)
        else:
            self.__dict__[key] = value
