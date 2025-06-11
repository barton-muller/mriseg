# This file contains the UNet model architecture as a PyTorch nn.Module class
import math

import torch
import torch.nn as nn
from .utils import pad_crop_tensor

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        """Initialize the UNetBlock"""
        super(UNetBlock, self).__init__()

        # Define the convolutional block
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """Forward pass"""
        return self.block(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, features=[64, 128, 256, 512]):
        """
        Initialize the UNet model.

        Args:
        in_channels : (int, optional)
            Number of input channels. Defaults to 1.
        out_channels : (int, optional)
            Number of output channels. Defaults to 4.
        features : (list, optional)
            Number of features in each layer. Defaults to [64, 128, 256, 512].
        """
        # Initialize the UNet
        super(UNet, self).__init__()

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder 
        enc_channels = [in_channels] + features
        self.encoders = nn.ModuleList([UNetBlock(enc_channels[i], enc_channels[i+1]) for i in range(len(features))])

        # Bottleneck
        self.bottleneck = UNetBlock(features[-1], features[-1] * 2)
        
        # build up-convolutions for decoder: double channels -> features[::-1]
        decoder_features = features[::-1]
        self.upconvs = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=(features[-1] * 2) if i == 0 else decoder_features[i],
                out_channels=decoder_features[i],
                kernel_size=2,
                stride=2,
            )
            for i in range(len(decoder_features))
        ])

        # build decoder blocks: concatenate skip connections -> feature maps
        self.decoders = nn.ModuleList([
            UNetBlock(
                in_channels=decoder_features[i] * 2,
                out_channels=(decoder_features[i+1] 
                            if (i+1) < len(decoder_features) 
                            else decoder_features[i])
            )
            for i in range(len(decoder_features))
        ])
        
        # final 1×1 conv to map to desired output channels
        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)

        # Making it backwards compatible 
        self.enc1, self.enc2, self.enc3, self.enc4 = self.encoders
        self.upconv4, self.upconv3, self.upconv2, self.upconv1 = self.upconvs
        self.dec4,   self.dec3,   self.dec2,   self.dec1   = self.decoders
        
    def forward(self, x):
        """Forward pass through the U-Net."""
        # Remember original size
        b, c, h0, w0 = x.shape

        # PRE-PAD
        # pad up to next multiple of 16 so all pool/upsample align
        h_tar = math.ceil(h0 / 16) * 16
        w_tar = math.ceil(w0 / 16) * 16
        x, pad_info, _ = pad_crop_tensor(x, (h_tar, w_tar))
        pad_left, pad_right, pad_top, pad_bottom = pad_info

        
        # Encoder pathway, collecting skip connections
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x) # pooling layer

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder pathway
        for i in range(len(self.upconvs)):
            x = self.upconvs[i](x)
            skip = skips[-(i+1)]
            # crop skip to x’s H×W here
            _, _, h, w = x.shape
            skip = skip[:, :, :h, :w]
            x = torch.cat([x, skip], dim=1)
            x = self.decoders[i](x)

        x = self.final(x)

        # POST‐CROP to original h0×w0
        # remove exactly what was padded
        y = x[:, :, pad_top : pad_top + h0, pad_left: pad_left + w0]
        return y
