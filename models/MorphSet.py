""" Full assembly of the parts to form the complete network """

import torch
import torch.nn as nn
import torch.nn.functional as F
from models import *
from .parts import *


class MorphSet(nn.Module):
    def __init__(self, n_channels, n_classes, enc):
        super(MorphSet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.enc = enc
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if True else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, True)
        self.up2 = Up(512, 256 // factor, True)
        self.up3 = Up(256, 128 // factor, True)
        self.up4 = Up(128, 64, True)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.enc(x)
        # torch.matmul(x1.permute(0,2,3,1).unsqueeze(3).repeat(1,1,1,8,1), torch.FloatTensor(8, 2048, 8).cuda()).shape

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits