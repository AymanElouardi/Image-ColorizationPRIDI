import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Block used in U-Net consisting of two consecutive convolution layers.
    Each convolution is followed by Batch Normalization and ReLU activation.
    """

    def __init__(self, inChannels, outChannels):
        """
        Args:
            inChannels (int): Number of input feature channels.
            outChannels (int): Number of output feature channels.
        """
        super().__init__()

        self.doubleConv = nn.Sequential(
            nn.Conv2d(inChannels, outChannels, kernel_size=3, padding=1),
            nn.BatchNorm2d(outChannels),
            nn.ReLU(inplace=True),

            nn.Conv2d(outChannels, outChannels, kernel_size=3, padding=1),
            nn.BatchNorm2d(outChannels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Forward pass of the double convolution block.
        """
        return self.doubleConv(x)


class UNet(nn.Module):
    """
    U-Net architecture for old picture colorization / restoration.

    Input:
        - Vintage degraded RGB image (3 channels)

    Output:
        - Restored RGB image (3 channels)
    """

    def __init__(self, inputChannels=3, outputChannels=3):
        """
        Args:
            inputChannels (int): Number of channels in input image (default: 3).
            outputChannels (int): Number of channels in output image (default: 3).
        """
        super().__init__()

        # ---------------- Encoder (Downsampling path) ----------------
        self.encoder1 = DoubleConv(inputChannels, 64)
        self.encoder2 = DoubleConv(64, 128)
        self.encoder3 = DoubleConv(128, 256)
        self.encoder4 = DoubleConv(256, 512)

        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ---------------- Decoder (Upsampling path) ----------------
        self.upConv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(512, 256)

        self.upConv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(256, 128)

        self.upConv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(128, 64)

        # ---------------- Output layer ----------------
        self.outputConv = nn.Conv2d(64, outputChannels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass of the U-Net.

        Args:
            x (Tensor): Input tensor of shape (B, 3, H, W)

        Returns:
            Tensor: Restored RGB image of shape (B, 3, H, W)
        """

        # -------- Encoder --------
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.maxPool(enc1))
        enc3 = self.encoder3(self.maxPool(enc2))
        enc4 = self.encoder4(self.maxPool(enc3))

        # -------- Decoder --------
        dec3 = self.upConv3(enc4)
        dec3 = torch.cat((enc3, dec3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upConv2(dec3)
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upConv1(dec2)
        dec1 = torch.cat((enc1, dec1), dim=1)
        dec1 = self.decoder1(dec1)

        output = self.outputConv(dec1)

        # Sigmoid ensures output values are in [0, 1]
        return torch.sigmoid(output)
