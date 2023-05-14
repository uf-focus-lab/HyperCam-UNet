# ---------------------------------------------------------
# Yuxuan Zhang
# Dept. of Electrical and Computer Engineering
# University of Florida
# ---------------------------------------------------------
# U_Net model implementation
# ---------------------------------------------------------
# PIP Modules
from typing import List
import torch
import torch.nn as nn
from torchvision import transforms

# Custom imports
from lib import Module, Context
from util.dataset import Sample_t
from .Generic import GenericModule

Features_T = List[torch.Tensor]


class U_Node(Module):
    def __init__(self, device, in_ch, out_ch):
        super().__init__(device=device)
        self.norm1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        # self.norm2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        # Random init
        for layer in [self.conv1, self.conv2]:
            nn.init.xavier_normal_(layer.weight)

    def forward(self, x):
        # out = self.norm1(x)
        out = self.conv1(x)
        out = self.relu1(out)
        # out = self.norm2(out)
        out = self.conv2(out)
        out = self.relu2(out)
        return out


class Encoder(Module):
    def __init__(self, ctx: Context, device, channels: List[int], sample: torch.Tensor):
        super().__init__(device=device)
        # Downscaler
        self.pool = nn.MaxPool2d((2, 2))
        # Initialize input sample
        nodes = []
        # Generate node list according to input sample and channels
        for c in channels:
            sample = self.pool(sample)
            # Get dimensions out of the current sample
            _, d, _, _ = sample.shape
            # Create new node layer using the sample
            layer = U_Node(device, d, c)
            # Iterate the sample
            sample = layer(sample)
            ctx.log("Encoder node shape", tuple(sample.shape))
            # Append layer to node list
            nodes.append(layer)
        # Instantiate node list
        self.nodes = nn.ModuleList(nodes)
        del nodes

    def forward(self, x) -> Features_T:
        features = []
        for node in self.nodes:
            x = self.pool(x)
            x = node(x)
            features.append(x)
        return features


class DecLayer(Module):
    def __init__(self, device, sample_x, sample_f, out_channels):
        super().__init__(device=device)
        # Upscale convolution
        b, c, h, w = sample_x.shape
        self.upconv = nn.ConvTranspose2d(c, out_channels, 2, 2)
        s = self.upconv(sample_x)
        # Scaler
        b, c, h, w = s.shape
        self.scaler = transforms.Resize((h, w), antialias=True)
        f = self.scaler(sample_f)
        # Concatenate
        s = torch.cat([s, f], dim=1)
        b, c, h, w = s.shape
        # Decoder node
        self.node = U_Node(device, c, out_channels)
        # Dropout
        self.dropout = nn.Dropout2d(p=0.5)

    def forward(self, x, f, train=None):
        x = self.upconv(x)
        f = self.scaler(f)
        x = torch.cat([x, f], dim=1)
        x = self.node(x)
        if train:
            x = self.dropout(x)
        return x


class Decoder(Module):
    def __init__(self, ctx: Context, device, channels: List[int], sample: Features_T):
        super().__init__(device=device)
        # Initialize parameters
        self.layer_count = len(channels)
        # Decompose packed tensors
        s = sample[-1]
        features = sample[: self.layer_count][::-1]
        layers = []
        # Generate layers
        for c, f in zip(channels, features):
            layer = DecLayer(device, s, f, c)
            s = layer(s, f)
            layers.append(layer)
            ctx.log(
                "Decoder node shape",
                tuple(s.shape),
                "|",
                f"ff. shape {tuple(f.shape)}",
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, features: Features_T, train=None):
        x = features[-1]
        features = features[: self.layer_count][::-1]
        for layer, f in zip(self.layers, features):
            x = layer(x, f, train=train)
        return x


class U_Net(GenericModule):
    def __init__(self, ctx: Context, device, sample: Sample_t):
        super().__init__(device=device)
        s, sample_out = sample
        ctx.log("model input shape", tuple(s.shape))
        # Optional Prescaler to match sample input's shape
        _, _, w, h = s.shape
        self.input_shape = (w, h)
        self.prescaler = transforms.Resize((w, h), antialias=True)
        # Encoder
        self.encoder = Encoder(ctx, device, [32, 128, 512, 1024], s)
        s = self.encoder(s)
        for i in range(len(s) - 1):
            ctx.log(f"Encoder feat. forward shape[{i}]", tuple(s[i].shape))
        ctx.log("Encoder output shape", tuple(s[-1].shape))
        # Decoder
        _, d, _, _ = sample_out.shape
        self.decoder = Decoder(ctx, device, [512, 400, d], s)
        s = self.decoder(s)
        # Report output shape
        ctx.log("Decoder output shape", tuple(s.shape))
        # Resize the decoder output to match sample output
        _, _, h, w = sample_out.shape
        self.scaler = transforms.Resize((h, w), antialias=True)
        s = self.scaler(s)
        ctx.log("Final output shape", tuple(s.shape))

    def forward(self, x, train=None):
        # x.shape = (Batches, Bands, Hight, Width)
        # optionally resize input to match sample input
        _, _, h, w = x.shape
        if (h, w) != self.input_shape:
            x = self.prescaler(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.scaler(x)
        return super().forward(x, train)
