import torch

class CustomLoss(torch.nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.mse = torch.nn.MSELoss()
        self.smoothL1 = torch.nn.SmoothL1Loss()

    def forward(self, pred, truth):
        # delta = (truth - pred).swapaxes(0, 1).contiguous()  # (d, b, w, h)
        # delta_bri = torch.abs(torch.mean(delta, dim=0))  # (b, w, h)
        # ratio = delta_bri / torch.max(delta_bri)  # (b, w, h)
        # # ae = torch.abs(delta)
        # se = torch.square(delta)
        # return torch.mean(se * (1 - ratio) + ratio**2)
        return self.smoothL1(pred, truth)
        # Punish the difference of overall hue difference
        delta_bands = torch.mean(pred, dim=(2, 3)) - torch.mean(truth, dim=(2, 3))
        delta_bands = torch.mean(torch.abs(delta_bands))

        # Punish the discontinuity of pixel across neighbor bands
        delta_pixel = pred[:, 1:, :, :] - pred[:, :-1, :, :]
        delta_pixel = torch.mean(torch.abs(delta_pixel))

        # Accumulate all parts of the loss
        return 1 * delta_bands + 1 * delta_pixel + 4 * self.mse(pred, truth)
