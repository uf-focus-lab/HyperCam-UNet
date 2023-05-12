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
import torch.nn as nn
from torchvision import transforms

# Custom imports
import cvtb
from lib import Module, Context
from util.augment import affine
from util.map_spectral import map_spectral, LED_LIST, REF_BANDS


class CustomLoss(Module):
    def __init__(self, device):
        super(CustomLoss, self).__init__(device=device)

    def forward(self, pred, truth):
        delta = (truth - pred).swapaxes(0, 1).contiguous()  # (d, b, w, h)
        delta_bri = torch.abs(torch.mean(delta, dim=0))  # (b, w, h)
        ratio = delta_bri / torch.max(delta_bri)  # (b, w, h)
        # ae = torch.abs(delta)
        se = torch.square(delta)
        return torch.mean(se * (1 - ratio) + ratio**2)


def rotate_axis(x: torch.Tensor):
    """(b, d, w, h) -> (b, w, h, d)"""
    x = x.swapaxes(1, 3)
    x = x.swapaxes(1, 2)
    return x.contiguous()


def reverse_axis(x: torch.Tensor):
    """(b, w, h, d) -> (b, d, w, h)"""
    x = x.swapaxes(1, 2)
    x = x.swapaxes(1, 3)
    return x.contiguous()


class GenericModule(Module):
    def __init__(self, device):
        super().__init__(device=device, loss=nn.HuberLoss().to(device))

    def preview(self, input, pred, truth):
        # Resize input to match prediction
        b, d, w, h = pred.shape
        input = transforms.Resize((w, h), antialias=True)(input)

        input, pred, truth = map(rotate_axis, [input, pred, truth])
        # Map spectra to RGB
        input = input[:, :, :, [1, 2, 6]]
        if d == len(REF_BANDS):
            pred = map_spectral(pred)
            truth = map_spectral(truth)
        else:
            assert d == len(LED_LIST), f"Invalid input dimension {d}"
            pred = pred[:, :, :, [1, 2, 6]]
            truth = truth[:, :, :, [1, 2, 6]]
        # Concatenate input and prediction into grids
        grid = [t.reshape(b * w, h, -1).swapaxes(0, 1) for t in [input, pred, truth]]
        grid = [t.cpu().numpy() for t in grid]
        grid = [cvtb.types.scaleToFit(t) for t in grid]
        return np.concatenate(grid, axis=0)

    def transform(self, ctx: Context, *data_point):
        batch, truth, *data_point = super().transform(ctx, *data_point)
        if ctx.train_mode:
            if "PRE_TRAIN" in ctx.train_mode:
                batch = map_spectral(rotate_axis(truth), range(len(LED_LIST)))
                batch = reverse_axis(batch)
            if "NARROW_BAND" in ctx.train_mode:
                truth = map_spectral(rotate_axis(truth), range(len(LED_LIST)))
                truth = reverse_axis(truth)
            if "AFFINE" in ctx.train_mode:
                batch, truth = affine(batch, truth)
        return batch, truth, *data_point

    def iterate_batch(self, ctx: Context, *data_point):
        if not ctx.train_mode:
            ctx.push("list", *data_point[2])
        return super().iterate_batch(ctx, *data_point)

    def forward(self, x: torch.Tensor, train=None):
        if train and "NARROW_BAND" in train:
            x = map_spectral(rotate_axis(x), range(len(LED_LIST)))
            x = reverse_axis(x)
        return x
