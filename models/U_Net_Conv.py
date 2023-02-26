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
    def __init__(self, in_ch, out_ch, **kwargs):
        super().__init__()
        # Set defaults
        if 'kernel_size' not in kwargs:
            kwargs['kernel_size'] = (3, 3)
        if 'padding' not in kwargs:
            kwargs['padding'] = (1, 1)
        if 'stride' not in kwargs:
            kwargs['stride'] = (1, 1)
        # self.norm1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, **kwargs)
        self.relu1 = nn.ReLU()
        # self.norm2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, **kwargs)
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
    def __init__(self, channels: List[int], sample: torch.Tensor):
        super().__init__()
        # Initialize layer array
        layers = []
        # Downscale mode
        INTP_MODE = transforms.InterpolationMode.BICUBIC
        # Generate node list according to input sample and channels
        for c in channels:
            # Get dimensions out of the current sample
            _, d, w, h = sample.shape
            # Downscaler
            # downscale = nn.MaxPool2d((2, 2))
            downscale = transforms.Resize(
                (int(w / 2), int(h / 2)),
                interpolation=INTP_MODE
            )
            sample = downscale(sample)
            # Create new node using the sample
            node = U_Node(d, c, kernel_size=3, padding=1)
            # Iterate the sample
            sample = node(sample)
            print("Encoder node shape", sample.shape)
            # Append node to list
            layers.append(nn.ModuleList([downscale, node]))
        # Instantiate node list
        self.layers = nn.ModuleList(layers)

    def forward(self, x) -> Features_T:
        features = []
        for downscale, node in self.layers:
            x = downscale(x)
            x = node(x)
            features.append(x)
        return features


class Decoder(nn.Module):
    def __init__(self, channels: List[int], sample: Features_T):
        super().__init__()
        # Initialize parameters
        self.layer_count = len(channels)
        # Decompose packed tensors
        sample = sample[::-1]
        offset = len(sample) - self.layer_count
        s, features = sample[0], sample[offset:]
        layers = []
        # Generate layers
        for i in range(len(channels)):
            _, d, _, _ = s.shape
            c = channels[i]
            # Upscale convolution
            upconv = nn.ConvTranspose2d(d, c, 2, 2)
            # Iterate input sample
            s = upconv(s)
            # Generate scaler
            _, _, w, h = s.shape
            scaler = transforms.Resize((w, h))
            f = scaler(features[i])
            s = torch.cat([s, f], dim=1)
            _, d, _, _ = s.shape
            # Concat sample with features
            node = U_Node(d, c)
            s: torch.Tensor = node(s)
            print("Decoder node shape", s.shape)
            # Register current layer
            layers.append(nn.ModuleList([upconv, scaler, node]))
        # Register all layers
        self.layers = nn.ModuleList(layers)

    def forward(self, features: Features_T):
        features = features[::-1]
        offset = len(features) - self.layer_count
        x, features = features[0], features[offset:]
        i = 0
        for upconv, scaler, node in self.layers:
            x = upconv(x)
            f = scaler(features[i]); features[i] = None
            x = torch.cat([x, f], dim=1)
            del f
            x = node(x)
            i += 1
        return x

class Model(Module):
    def __init__(self, device, sample: Sample_t):
        super(Model, self).__init__(device)
        s, sample_out = sample
        print("model input shape", s.shape)
        # Encoder
        self.encoder = Encoder([32, 128, 512, 2048], s)
        s = self.encoder(s)
        # Decoder
        self.decoder = Decoder([512, 128, 32], s)
        s = self.decoder(s)
        # Resize the decoder output to match sample output
        _, _, w, h = sample_out.shape
        self.scaler = transforms.Resize((w, h))
        s = self.scaler(s)
        # FC Layers to Complete Spectrum Information
        out_channels = int(sample_out.shape[1])
        _, c, _, _ = s.shape
        fc_layers = []
        while c < out_channels:
            _c = c
            c = c * 2 if c * 2 <= out_channels else out_channels
            fc_layers.append(nn.Conv2d(_c, c, kernel_size=1))
        self.fc = nn.Sequential(*fc_layers)
        s = self.fc(s)
        # Report output shape
        print("Final result shape", s.shape)

    def forward(self, x):
        # x.shape = (Batches, Bands, Hight, Width)
        bri_map = torch.stack((torch.mean(x, dim=1),), dim=1)
        out = x / bri_map
        # Learnable layers
        out = self.encoder(out)
        out = self.decoder(out)
        out = self.scaler(out)
        out = self.fc(out)
        # out = self.sigmoid(out)
        bri_map = self.scaler(bri_map)
        bri_map.detach()
        return out, bri_map
