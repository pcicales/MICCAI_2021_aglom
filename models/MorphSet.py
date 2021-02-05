""" Full assembly of the parts to form the complete network """
from config import options
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import *
from .parts import *

# If you want to test if splits are behaving as you want:
# torch.split(x, 8)[0] == x.view(3, 8, 8192, 4, 4)[0]

class MorphSet(nn.Module):
    def __init__(self, n_channels, n_classes, enc):
        super(MorphSet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        if options.encoder == 'resnet50':
            self.inchans = 2048

        self.enc = enc
        self.SE1 = SE_Block(self.inchans)
        self.preset = Convk1(in_channels=self.inchans, out_channels=options.preset_channels)
        self.SE2 = SE_Block(options.preset_channels)
        self.setformer = ConvSet(A=1, B=options.set_points, K=3, P=int(options.preset_channels**0.5),
                                 stride=2, pad=1, heads=options.heads)
        self.SE3 = SE_Block(options.preset_channels * options.set_points)
        self.postset = Convk1(in_channels=options.preset_channels * options.set_points,
                              out_channels=options.postset_channels)
        self.SE4 = SE_Block(options.postset_channels)
        self.fpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = Out(in_channels=options.postset_channels * options.stack_size, out_channels=options.num_classes)

    def forward(self, x):
        x = self.enc(x)
        x = self.SE1(x)
        x = self.preset(x)
        x = self.SE2(x)
        # x.shape = b, c, h, w
        x = self.setformer(x)
        x = self.SE3(x)
        # x.shape = b, c*set, h1, w1
        x = self.postset(x)
        x = self.SE4(x)
        x = x.reshape(len(x) // options.stack_size, options.stack_size * x.shape[1], x.shape[2], x.shape[3]).contiguous()
        x = self.fpool(x)
        x = self.classifier(x)
        return x