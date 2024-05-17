# ---------------------------------------------------------
# Yuxuan Zhang
# Dept. of Electrical and Computer Engineering
# University of Florida
# ---------------------------------------------------------
# Generic model for spectrum reconstruction tasks
# ---------------------------------------------------------
# PIP Modules
import numpy as np
import torch
from torchvision import transforms
# Custom imports
import cvtb
from lib import Module, Context
from util.device import DEVICE
from util.augment import affine
from util.map_spectral import map_spectral, LED_LIST, REF_BANDS


def rotate_axis(x: torch.Tensor):
    """(b, d, w, h) -> (b, w, h, d)"""
    x = x.swapaxes(1, 3)
    x = x.swapaxes(1, 2)
    return x.contiguous()


class GenericModule(Module):
    def __init__(self, device):
        super().__init__(device=device)

    def preview(self, input, pred, truth):
        # Resize input to match prediction
        b, d, w, h = pred.shape
        input = transforms.Resize((w, h), antialias=True)(input)
        # Map spectra to RGB
        input = input[:, [1, 2, 6]]
        if d == len(REF_BANDS):
            pred = map_spectral(pred)
            truth = map_spectral(truth)
        else:
            assert d == len(LED_LIST), f"Invalid input dimension {d}"
            pred = pred[:, :, :, [1, 2, 6]]
            truth = truth[:, :, :, [1, 2, 6]]
        # Rotate axis
        input, pred, truth = map(rotate_axis, [input, pred, truth])
        # Concatenate input and prediction into grids
        grid = [t.reshape(b * w, h, -1).swapaxes(0, 1) for t in [input, pred, truth]]
        grid = [t.cpu().numpy() for t in grid]
        # return cvtb.types.scaleToFit(np.concatenate(grid, axis=0))
        
        grid = [cvtb.types.scaleToFit(t) for t in grid]
        return np.concatenate(grid, axis=0)

    def transform(self, ctx: Context, *data_point):
        batch, truth, *data_point = super().transform(ctx, *data_point)
        if ctx.train_mode:
            if "PRE_TRAIN" in ctx.train_mode:
                del batch
                batch = map_spectral(truth, range(len(LED_LIST)))
            if "NARROW_BAND" in ctx.train_mode:
                truth = map_spectral(truth, range(len(LED_LIST)))
            if "AFFINE" in ctx.train_mode:
                for i in range(len(batch)):
                    batch[i:i+1], truth[i:i+1] = affine(batch[i:i+1], truth[i:i+1])
            if "MIX_IN" in ctx.train_mode:
                mix = map_spectral(truth, range(len(LED_LIST)))
                mix = transforms.Resize(list(batch.shape[-2:]), antialias=True)(mix)
                mix[0] = batch[0]
                mix[-1] = batch[-1]
                batch = mix
        return batch, truth, *data_point

    def iterate_batch(self, ctx: Context, *data_point):
        if not ctx.train_mode:
            ctx.push("list", *data_point[2])
        return super().iterate_batch(ctx, *data_point)

    def forward(self, x: torch.Tensor, train=None):
        if train and "NARROW_BAND" in train:
            x = map_spectral(x, range(len(LED_LIST)))
            x = x
        return x
