# ---------------------------------------------------------
# Yuxuan Zhang
# Dept. of Electrical and Computer Engineering
# University of Florida
# ---------------------------------------------------------
# Data preprocessing methods
# ---------------------------------------------------------
# PIP Modules
import torch

def __otsu_threshold(hsi: torch.Tensor, bins=1000) -> torch.Tensor:
    d, w, h = hsi.shape
    # Compute histogram
    hist = [torch.histc(hsi[i], bins=bins, min=0, max=1) for i in range(d)]
    hist = torch.stack(hist, dim=0) / (w * h)

    # Compute cumulative sums
    bins_idx = torch.linspace(0, 1, bins).to(hsi.device)
    cumulative_sum = torch.cumsum(hist, dim=-1)
    cumulative_mean = torch.cumsum(hist * bins_idx, dim=-1)

    # Compute global mean
    global_mean = cumulative_mean[..., -1:]

    # Compute between-class variance
    bc_variance = (global_mean * cumulative_sum - cumulative_mean) ** 2 / (
        cumulative_sum * (1 - cumulative_sum + 1e-10)
    )

    # Handle NaN/inf values (these occur when there's only one class of pixels in the image)
    bc_variance[cumulative_sum == 0] = 0
    bc_variance[cumulative_sum == 1] = 0

    # Find threshold
    thresholds = torch.argmax(bc_variance, dim=-1)
    # Convert bin indices to values in the range [0, 1]
    thresholds = thresholds.float() / bins

    return thresholds.view(d, 1, 1)

def remove_spots(hsi: torch.Tensor) -> torch.Tensor:
    threshold = __otsu_threshold(hsi)
    # Re-equalize HSI image
    hsi = hsi / threshold
    # Drive mask from thresholds
    # mask = 4 * (1 - hsi)
    # mask = torch.clamp(1 - mask, 0, 1)
    # return mask
    mask = torch.ones(hsi.shape, dtype=hsi.dtype).to(hsi.device)
    mask[hsi > 1] = 0.01
    # assert not (mask <= 0).any(), "Zero or negative values detected in mask"
    # Replace infrared mask with red mask
    mask[7] = mask[6]
    # Apply float mask to HSI
    hsi = hsi * mask
    # Fix the masked pixels with the average of the remainder pixels
    remainder = torch.sum(hsi, dim=0) / torch.sum(mask, dim=0)
    # assert not torch.isnan(remainder).any(), "NaN values detected in remainder"
    hsi += remainder * (1 - mask)
    # Re-equalize HSI image
    hsi = hsi * threshold
    return hsi.contiguous()


def _remove_spots(hsi: torch.Tensor) -> torch.Tensor:
    d, w, h = hsi.shape
    # Equalize the histogram of each layer
    eq = torch.mean(hsi, dim=(1, 2)).view(d, 1, 1)
    hsi = hsi / eq
    # Select the brightest and second brightest pixels
    idx = torch.argsort(hsi, dim=0, descending=True)
    peak = (idx[0], idx[1])
    grid_w, grid_h = torch.meshgrid(torch.arange(w), torch.arange(h))
    p_layers = [hsi[p, grid_w, grid_h] for p in peak]
    # Drive the remainder pixels' average
    remainder = (torch.sum(hsi, dim=0) - p_layers[0]) / (d - 1)
    # Select pixels that are more than 150% as bright as the second brightest pixel
    # These pixels are considered as LED light spots
    mask = (p_layers[0] > 2 * p_layers[1])
    print(float(torch.sum(p_layers[0]) / torch.sum(p_layers[1])))
    # Assign the remainder pixels' average to selected pixels
    hsi[peak[0], grid_w, grid_h][mask] = remainder[mask]
    # Re-apply the equalization
    hsi = hsi * eq
    # Clean up memory
    del idx, remainder, mask, p_layers, peak, grid_w, grid_h
    return hsi.contiguous()
