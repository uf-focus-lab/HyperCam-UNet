# ---------------------------------------------------------
# Yuxuan Zhang
# Dept. of Electrical and Computer Engineering
# University of Florida
# ---------------------------------------------------------
# Pipeline of convolutional layers and activation functions
# ---------------------------------------------------------
# PIP Modules
import torch.nn as nn
from torchvision import transforms
import numpy as np

# Custom Modules
from lib import Module, Context
from util.dataset import Sample_t
from .Generic import GenericModule

DROP_RATE = 0.1
RAND_MEAN = 0
RAND_STD = 1e-8


class ConvPipe(Module):
    def __init__(
        self, in_ch, out_ch, layers=2, activation=nn.LeakyReLU, kernel_size=3, padding=1
    ):
        super().__init__()
        self.norm = nn.BatchNorm2d(in_ch)
        layer_list = []
        channels = np.array(range(layers + 1), np.float32) / layers
        channels = np.rint(channels * (out_ch - in_ch) + in_ch).astype(np.int32)
        for i in range(layers):
            conv = nn.Conv2d(
                channels[i], channels[i + 1], kernel_size=kernel_size, padding=padding
            )
            nn.init.xavier_uniform_(conv.weight)
            layer_list.append(conv)
            layer_list.append(activation())
        self.pipe = nn.Sequential(*layer_list)
        self.dropout = nn.Dropout2d(DROP_RATE)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x, train=None):
        if train:
            x = self.norm(x)
        x = self.pipe(x)
        if train:
            x = self.dropout(x)
        # x = self.sigmoid(x)
        return x


class ExpandPipe(Module):
    def __init__(self, in_ch, out_ch, layers=2, activation=nn.LeakyReLU):
        super().__init__()
        self.norm = nn.BatchNorm2d(in_ch)
        layer_list = []
        channels = np.array(range(layers + 1), np.float32) / layers
        channels = np.rint(channels * (out_ch - in_ch) + in_ch).astype(np.int32)
        for i in range(layers):
            linear = nn.Linear(channels[i], channels[i + 1])
            nn.init.xavier_uniform_(linear.weight)
            layer_list.append(linear)
            layer_list.append(activation())
        self.pipe = nn.Sequential(*layer_list)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x, train=None):
        b, c, w, h = x.shape
        if train:
            x = self.norm(x)
        x = x.swapaxes(1, 3)  # (b, h, w, c)
        x = self.pipe(x)
        x = x.swapaxes(1, 3)  # (b, c, w, h)
        # x = self.sigmoid(x)
        return x


class Conv(GenericModule):
    def __init__(self, ctx: Context, device, sample: Sample_t, c_mid=64):
        super().__init__(device=device)
        s_in, s_out = sample
        b, c_in, w_in, h_in = s_in.shape
        b, c_out, w_out, h_out = s_out.shape
        self.front = ConvPipe(c_in, c_mid, layers=8)
        self.resize = transforms.Resize((w_out, h_out))
        self.back = ConvPipe(c_mid, c_mid, layers=8)
        self.expand = ExpandPipe(c_mid, c_out, layers=8)

    def forward(self, x, train=None):
        x = self.front(x, train=train)
        x = self.resize(x)
        x = self.back(x, train=train)
        x = self.expand(x, train=train)
        return super().forward(x, train=train)
