import numpy as np
from torch import Tensor, mean, square
import torch.nn as nn

def score(pred: Tensor, truth: Tensor) -> str:
	avgErr = float(mean((pred - truth).abs()).detach().cpu())
	stdev = float(mean(square(pred - truth)).detach().cpu())
	return avgErr, stdev
