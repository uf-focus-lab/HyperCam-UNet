import numpy as np
from torch import Tensor, mean, square, sqrt
import torch.nn as nn

def score(pred: Tensor, truth: Tensor) -> tuple((Tensor, Tensor)):
	diff = (pred - truth).view((-1))
	#sqrt(sum(diff.^2))
	avgErr = mean(diff).abs()
	stdev = sqrt(mean(square(diff)))
	return avgErr.detach().cpu(), stdev.detach().cpu()
