from module import Module
from typing import List
import torch
import torch.nn as nn
from torchvision import transforms
from dataset import Sample_t
from env import DEVICE

# lossFunction = nn.CrossEntropyLoss().to(DEVICE)
lossFunction = nn.BCEWithLogitsLoss().to(DEVICE)

Features_T = List[torch.Tensor]

class U_Node(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.norm = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.relu1 = nn.LeakyReLU()
        # self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        # self.relu2 = nn.ReLU()

    def forward(self, x):
        out = self.norm(x)
        out = self.conv1(out)
        out = self.relu1(out)
        # out = self.conv2(out)
        # out = self.relu2(out)
        return out


class Encoder(nn.Module):
    def __init__(self, sample: Sample_t, channels = [16, 32, 64, 128, 256, 512, 1024]):
        super().__init__()
        # Downscaler
        self.pool = nn.MaxPool2d((2, 2))
        # Initialize input sample
        s, _ = sample
        nodes = []
        print("Encoder input shape", s.shape)
        # Generate node list according to input sample and channels
        for c in channels:
            s = self.pool(s)
            # Get dimensions out of the current sample
            _, d, _, _ = s.shape
            # Create new node layer using the sample
            layer = U_Node(d, c)
            # Iterate the sample
            s = layer(s)
            print("Encoder node shape", s.shape)
            # Append layer to node list
            nodes.append(layer)
        # Instantiate node list
        self.nodes = nn.ModuleList(nodes)

    def forward(self, x) -> Features_T:
        features = []
        for node in self.nodes:
            x = self.pool(x)
            x = node(x)
            features.append(x)
        return features


class Decoder(nn.Module):
    def __init__(self, sample: tuple[Features_T, torch.Tensor], channels=[1000, 800, 600, 400]):
        super().__init__()
        # Initialize samples
        s_in, s_out = sample
        s_in = s_in[::-1]
        s, features = s_in[0], s_in[1:]
        print("Decoder input shape", s.shape)
        # Initialize parameters
        channels.append(s_out.shape[1])
        self.layer_count = len(channels)
        upconvs = []
        dec_nodes = []
        # Generate layers
        for i in range(len(channels)):
            _, d, _, _ = s.shape
            c = channels[i]
            # Upscale convolution
            upconv = nn.ConvTranspose2d(d, c, 2, 2)
            upconvs.append(upconv)
            # Iterate input sample
            s: torch.Tensor = upconv(s)
            s = torch.cat([s, features[i]], dim=1)
            _, d, _, _ = s.shape
            # Concat sample with features
            decoder = U_Node(d, c)
            dec_nodes.append(decoder)
            s: torch.Tensor = decoder(s)
            print("Decoder node shape", s.shape)

        self.upconvs = nn.ModuleList(upconvs)
        self.dec_nodes = nn.ModuleList(dec_nodes)

    def forward(self, features: Features_T):
        features = features[::-1]
        x, features = features[0], features[1:]
        for i in range(self.layer_count):
            x = self.upconvs[i](x)
            x = torch.cat([x, features[i]], dim=1)
            x = self.dec_nodes[i](x)
        return x

class Model(Module):
    def __init__(self, device, sample: Sample_t):
        super(Model, self).__init__(device)
        sample_in, sample_out = sample
        # Encoder
        self.encoder = Encoder(sample)
        sample_mid = self.encoder(sample_in)
        # Decoder
        self.decoder = Decoder((sample_mid, sample_out))
        s = self.decoder(sample_mid)
        # Report output shape
        print("Decoder result shape", s.shape)
        # Resize the decoder output to match sample output
        _, _, w, h = sample_out.shape
        self.scaler = transforms.Resize((w, h))
        s = self.scaler(s)
        # self.sigmoid = nn.Sigmoid()
        # s = self.sigmoid(s)
        print("Final result shape", s.shape)

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        out = self.scaler(out)
        # out = self.sigmoid(out)
        return out
