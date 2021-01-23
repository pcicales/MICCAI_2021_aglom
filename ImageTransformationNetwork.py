import torch
import torch.nn as nn
import torch.nn.functional as F
from config import options
import random

#############
# StyPath ###
#############


class SEStyEncNet(torch.nn.Module):
    def __init__(self, num_styles):
        super(SEStyEncNet, self).__init__()
        self.conv_1 = CondConvolution(3, 32, 3, 1, (1, 1), num_styles)
        self.se1 = SE_Block(32)
        self.conv_2 = CondConvolution(32, 64, 3, 1, (2, 2), num_styles)
        self.se2 = SE_Block(64)
        self.conv_3 = CondConvolution(64, 128, 3, 1, (2, 2), num_styles)
        self.se3 = SE_Block(128)

        self.res_block1 = ResBlock(num_styles)
        self.se4 = SE_Block(128)
        self.res_block2 = ResBlock(num_styles)
        self.se5 = SE_Block(128)

        # self.upsample1 = Upsampling(128, 64, 3, 1, (1, 1), num_styles)
        # self.se6 = SE_Block(64)
        # self.upsample2 = Upsampling(64, 32, 3, 1, (1, 1), num_styles)
        # self.se7 = SE_Block(32)

        self.upsample1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2, num_styles=num_styles)
        self.se6 = SE_Block(64)
        self.upsample2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2, num_styles=num_styles)
        self.se7 = SE_Block(32)

        self.conv_4 = CondConvolution(32, 3, 3, 1, (1, 1), num_styles, False)

    def forward(self, x, style_no, style_no2=None, style_no3=None, alphas=None):
        x = self.conv_1(x, style_no, style_no2, style_no3, alphas)
        x = self.se1(x)
        x = self.conv_2(x, style_no, style_no2, style_no3, alphas)
        x = self.se2(x)
        x = self.conv_3(x, style_no, style_no2, style_no3, alphas)
        x = self.se3(x)
        x = self.res_block1(x, style_no, style_no2, style_no3, alphas)
        x = self.se4(x)
        x = self.res_block2(x, style_no, style_no2, style_no3, alphas)
        x = self.se5(x)
        x = self.upsample1(x, style_no, style_no2, style_no3, alphas)
        x = self.se6(x)
        x = self.upsample2(x, style_no, style_no2, style_no3, alphas)
        x = self.se7(x)
        x = self.conv_4(x, style_no, style_no2, style_no3, alphas)
        x = F.sigmoid(x)
        return x


class SE_Block(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)


class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None, num_styles=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = CondConvolution(in_channels, out_channels, kernel_size, 0, stride, num_styles)

    def forward(self, x, style_no, style_no2=None, style_no3=None, alphas=None):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out, style_no, style_no2, style_no3, alphas)
        return out


class CondConvolution(nn.Module):
    def __init__(self, input_filters, output_filters, kernel_size, padding, stride, num_styles, act=True):
        super(CondConvolution, self).__init__()
        self.reflection2d = nn.ReflectionPad2d(padding)
        self.conv = nn.Conv2d(input_filters, output_filters, kernel_size=kernel_size, stride=stride)
        self.instnorm = nn.InstanceNorm2d(output_filters, affine=True)
        self.act = act

        # Conditional Instance Parameters - Allows for transferring of multiple styles
        self.gamma = torch.nn.Parameter(data=torch.Tensor(num_styles, output_filters), requires_grad=True)
        nn.init.normal_(self.gamma, mean=0.0, std=0.1)  # 0.01
        self.beta = torch.nn.Parameter(data=torch.Tensor(num_styles, output_filters), requires_grad=True)
        nn.init.normal_(self.beta, mean=0.0, std=0.1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
            elif isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, style_no1, style_no2=None, style_no3=None, alphas=None):
        x = self.reflection2d(x)
        x = self.conv(x)
        x = self.instnorm(x)
        b, d, w, h = x.size()
        x = x.view(b, d, w * h)

        # Index into gamma and beta parameters to select specific style
        if options.mix_styles:
            gamma = alphas[0] * self.gamma[style_no1] + alphas[1] * self.gamma[style_no2] + \
                alphas[2] * self.gamma[style_no3] # gamma is N by C
            beta = alphas[0] * self.beta[style_no1] + alphas[1] * self.beta[style_no2] + \
                alphas[2] * self.beta[style_no3]
        else:
            gamma = self.gamma[style_no1]
            beta = self.beta[style_no1]
        x = (x * gamma.unsqueeze(-1).expand_as(x) + beta.unsqueeze(-1).expand_as(x)).view(b, d, w, h)

        if self.act:
            x = F.relu(x)
        return x

#############
# Vanilla ###
#############


class TransformerNet(torch.nn.Module):
    def __init__(self, num_styles):
        super(TransformerNet, self).__init__()
        self.conv_1 = CondConvolution(3, 32, 9, 4, (1, 1), num_styles)
        self.conv_2 = CondConvolution(32, 64, 3, 1, (2, 2), num_styles)
        self.conv_3 = CondConvolution(64, 128, 3, 1, (2, 2), num_styles)

        self.res_block1 = ResBlock(num_styles)
        self.res_block2 = ResBlock(num_styles)
        self.res_block3 = ResBlock(num_styles)
        self.res_block4 = ResBlock(num_styles)

        self.upsample1 = Upsampling(128, 64, 3, 1, (1, 1), num_styles)
        self.upsample2 = Upsampling(64, 32, 3, 1, (1, 1), num_styles)

        self.conv_4 = CondConvolution(32, 3, 9, 4, (1, 1), num_styles, False)

    def forward(self, x, style_no):
        x = self.conv_1(x, style_no)
        x = self.conv_2(x, style_no)
        x = self.conv_3(x, style_no)
        x = self.res_block1(x, style_no)
        x = self.res_block2(x, style_no)
        x = self.res_block3(x, style_no)
        x = self.res_block4(x, style_no)
        x = self.upsample1(x, style_no)
        x = self.upsample2(x, style_no)
        x = self.conv_4(x, style_no)
        x = F.sigmoid(x)
        return x


"""class CondConvolution(nn.Module):
    def __init__(self, input_filters, output_filters, kernel_size, padding, stride, num_styles, act=True):
        super(CondConvolution, self).__init__()
        self.reflection2d = nn.ReflectionPad2d(padding)
        self.conv = nn.Conv2d(input_filters, output_filters, kernel_size=kernel_size, stride=stride)
        self.instnorm = nn.InstanceNorm2d(output_filters, affine=True)
        self.act = act

        # Conditional Instance Parameters - Allows for transferring of multiple styles
        self.gamma = torch.nn.Parameter(data=torch.Tensor(num_styles, output_filters), requires_grad=True)
        nn.init.normal_(self.gamma, mean=0.0, std=0.1)  # 0.01
        self.beta = torch.nn.Parameter(data=torch.Tensor(num_styles, output_filters), requires_grad=True)
        nn.init.normal_(self.beta, mean=0.0, std=0.1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
            elif isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, style_no):
        x = self.reflection2d(x)
        x = self.conv(x)
        x = self.instnorm(x)
        b, d, w, h = x.size()
        x = x.view(b, d, w * h)

        # Index into gamma and beta parameters to select specific style
        gamma = self.gamma[style_no]  # gamma is N by C
        beta = self.beta[style_no]
        x = (x * gamma.unsqueeze(-1).expand_as(x) + beta.unsqueeze(-1).expand_as(x)).view(b, d, w, h)

        if self.act:
            x = F.relu(x)
        return x"""


class ResBlock(nn.Module):
    def __init__(self, num_styles):
        super(ResBlock, self).__init__()
        self.res_conv1 = CondConvolution(128, 128, 3, 1, (1, 1), num_styles)
        self.res_conv2 = CondConvolution(128, 128, 3, 1, (1, 1), num_styles, False)

    def forward(self, x, style_no, style_no2=None, style_no3=None, alphas=None):
        residual = x
        out = self.res_conv1(x, style_no, style_no2, style_no3, alphas)
        out = self.res_conv2(out, style_no, style_no2, style_no3, alphas)
        out += residual
        return out


class Upsampling(nn.Module):
    def __init__(self, input_filters, output_filters, kernel_size, padding, stride, num_styles):
        super(Upsampling, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = CondConvolution(input_filters, output_filters, kernel_size, padding, stride, num_styles)

    def forward(self, x, style_no, style_no2=None, style_no3=None, alphas=None):
        x = self.upsample(x)
        x = self.conv(x, style_no, style_no2, style_no3, alphas)
        return x
