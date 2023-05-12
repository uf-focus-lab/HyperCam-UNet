# ---------------------------------------------------------
# Yuxuan Zhang
# Dept. of Electrical and Computer Engineering
# University of Florida
# ---------------------------------------------------------
# TODO: Add description
# ---------------------------------------------------------
# PIP Modules
import torch
import numpy as np

# Custom Modules
import cvtb
from util.param import LED_LIST, REF_BANDS
from util.device import DEVICE

# Interpolate spectral raw
weights = []
x = np.array(REF_BANDS)
for name, bandwidth, delta in LED_LIST:
    gaussian = cvtb.fx.gaussian(bandwidth, delta / 20)
    gaussian = torch.Tensor(gaussian(x))
    gaussian = gaussian / torch.sum(gaussian)
    weights.append(gaussian.to(DEVICE))


def map_spectral(img: torch.Tensor, leds=[1, 2, 6]):
    assert img.device == DEVICE
    dim = img.ndim - 1
    img = [torch.sum(img * weights[i], dim=dim) for i in leds]
    img = torch.stack(img, dim=dim)
    return img
