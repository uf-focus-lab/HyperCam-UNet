import numpy as np
from torch import Tensor, mean, square, sqrt
import torch.nn as nn

def score(pred: Tensor, truth: Tensor, detach: bool = True) -> str:
	diff = (pred - truth).view((-1))
	#sqrt(sum(diff.^2))
	avgErr = mean(diff).abs()
	stdev = sqrt(mean(square(diff)))
	if detach:
		avgErr = float(avgErr.detach().cpu())
		stdev = float(stdev.detach().cpu())
	return avgErr, stdev
