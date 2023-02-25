from module import Module
from typing import List, Tuple
import torch
import torch.nn as nn
from torchvision import transforms
from dataset import Sample_t
from env import DEVICE

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, pred, truth):
        return torch.sum(torch.abs(truth - pred))

# lossFunction = nn.BCEWithLogitsLoss().to(DEVICE)
# lossFunction = nn.CrossEntropyLoss().to(DEVICE)
lossFunction = CustomLoss().to(DEVICE)

Features_T = List[torch.Tensor]

class U_Node(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # self.norm1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        # self.norm2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        # Random init
        for layer in [self.conv1, self.conv2]:
            nn.init.normal_(layer.weight, mean=0, std=1e-4)

    def forward(self, x):
        # out = self.norm1(x)
        out = self.conv1(x)
        out = self.relu1(out)
        # out = self.norm2(out)
        out = self.conv2(out)
        out = self.relu2(out)
        return out


class Encoder(nn.Module):
    def __init__(self, sample: Sample_t, channels = [128, 256, 512, 1024]):
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
    def __init__(self, sample: Tuple[Features_T, torch.Tensor], channels=[800, 600, 300]):
        super().__init__()
        # Initialize samples
        s_in, s_out = sample
        # Initialize parameters
        channels.append(s_out.shape[1])
        self.layer_count = len(channels)
        # Decompose packed tensors
        s_in = s_in[::-1]
        offset = len(s_in) - self.layer_count
        s, features = s_in[0], s_in[offset:]
        print("Decoder input shape", s.shape)
        upconvs = []
        scalers = []
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
            # Generate scaler
            _, _, w, h = s.shape
            scaler = transforms.Resize((w, h))
            scalers.append(scaler)
            f = scaler(features[i])
            s = torch.cat([s, f], dim=1)
            _, d, _, _ = s.shape
            # Concat sample with features
            decoder = U_Node(d, c)
            dec_nodes.append(decoder)
            s: torch.Tensor = decoder(s)
            print("Decoder node shape", s.shape)

        self.upconvs = nn.ModuleList(upconvs)
        self.scalers = nn.ModuleList(scalers)
        self.dec_nodes = nn.ModuleList(dec_nodes)

    def forward(self, features: Features_T):
        features = features[::-1]
        offset = len(features) - self.layer_count
        x, features = features[0], features[offset:]
        for i in range(self.layer_count):
            x = self.upconvs[i](x)
            f = self.scalers[i](features[i])
            x = torch.cat([x, f], dim=1)
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
        # x.shape = (Batches, Bands, Hight, Width)
        bri_map = torch.stack((torch.mean(x, dim=1),), dim=1)
        out = x / bri_map
        # Learnable layers
        out = self.encoder(out)
        out = self.decoder(out)
        out = self.scaler(out)
        # out = self.sigmoid(out)
        return out, self.scaler(bri_map)
