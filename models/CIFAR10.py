import torch
from torch import nn

import separableconv.nn as nn_sep

def make_separable_conv(
        in_channels, out_channels, kernel_size=3, stride=1, padding=0
    ):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels,
            ),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )


class CIFAR300K(nn.Module):
    def __init__(self):
        super(CIFAR300K, self).__init__()

        self.conv1 = nn_sep.SeparableConv2d(in_channels=3, out_channels=64, kernel_size = 3, stride=1, padding="same")#, normalization_dw=None, normalization_pw=None)
        self.conv2 = nn_sep.SeparableConv2d(in_channels=64, out_channels=64, kernel_size = 3, stride=1, padding="same")#, normalization_dw=None, normalization_pw=None), normalization_dw=None, normalization_pw=None)

        self.conv3 = nn_sep.SeparableConv2d(in_channels=64, out_channels=128, kernel_size = 3, stride=2, padding="valid")#, normalization_dw=None, normalization_pw=None), normalization_dw=None, normalization_pw=None)
        self.conv4 = nn_sep.SeparableConv2d(in_channels=128, out_channels=128, kernel_size = 3, stride=1, padding="same")#, normalization_dw=None, normalization_pw=None), normalization_dw=None, normalization_pw=None)
        self.conv5 = nn_sep.SeparableConv2d(in_channels=128, out_channels=128, kernel_size = 3, stride=1, padding="same")#, normalization_dw=None, normalization_pw=None), normalization_dw=None, normalization_pw=None)

        self.conv6 = nn_sep.SeparableConv2d(in_channels=128, out_channels=256, kernel_size = 3, stride=2, padding="valid")#, normalization_dw=None, normalization_pw=None), normalization_dw=None, normalization_pw=None)
        self.conv7 = nn_sep.SeparableConv2d(in_channels=256, out_channels=256, kernel_size = 3, stride=1, padding="same")#, normalization_dw=None, normalization_pw=None), normalization_dw=None, normalization_pw=None)
        self.conv8 = nn_sep.SeparableConv2d(in_channels=256, out_channels=256, kernel_size = 3, stride=1, padding="same")#, normalization_dw=None, normalization_pw=None), normalization_dw=None, normalization_pw=None)

        self.conv9 = nn_sep.SeparableConv2d(in_channels=256, out_channels=512, kernel_size = 3, stride=2, padding="valid")#, normalization_dw=None, normalization_pw=None), normalization_dw=None, normalization_pw=None)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.norm = nn.BatchNorm1d(512, affine=False)
        self.fc = nn.Linear(512, 10)
        # All activations are ReLU
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act((self.conv1(x)))
        # print(x.size())
        x = self.act((self.conv2(x)))
        # print(x.size())
        x = self.act((self.conv3(x)))
        # print(x.size())
        x = self.act((self.conv4(x)))
        # print(x.size())
        x = self.act((self.conv5(x)))
        # print(x.size())
        x = self.act((self.conv6(x)))
        # print(x.size())
        x = self.act((self.conv7(x)))
        # print(x.size())
        x = self.act((self.conv8(x)))
        # print(x.size())
        x = self.act((self.conv9(x)))


        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        # x = self.norm(x)
        x = self.fc(x)
        return x


class CIFAR900K(nn.Module):
    def __init__(self):
        super(CIFAR900K, self).__init__()

        self.conv1 = nn_sep.SeparableConv2d(in_channels=3, out_channels=64, kernel_size = 3, stride=1, padding="same")#, normalization_dw=None, normalization_pw=None)
        self.conv2 = nn_sep.SeparableConv2d(in_channels=64, out_channels=64, kernel_size = 3, stride=1, padding="same")#, normalization_dw=None, normalization_pw=None), normalization_dw=None, normalization_pw=None)

        self.conv3 = nn_sep.SeparableConv2d(in_channels=64, out_channels=128, kernel_size = 3, stride=2, padding="valid")#, normalization_dw=None, normalization_pw=None), normalization_dw=None, normalization_pw=None)
        self.conv4 = nn_sep.SeparableConv2d(in_channels=128, out_channels=128, kernel_size = 3, stride=1, padding="same")#, normalization_dw=None, normalization_pw=None), normalization_dw=None, normalization_pw=None)
        self.conv5 = nn_sep.SeparableConv2d(in_channels=128, out_channels=128, kernel_size = 3, stride=1, padding="same")#, normalization_dw=None, normalization_pw=None), normalization_dw=None, normalization_pw=None)

        self.conv6 = nn_sep.SeparableConv2d(in_channels=128, out_channels=256, kernel_size = 3, stride=2, padding="valid")#, normalization_dw=None, normalization_pw=None), normalization_dw=None, normalization_pw=None)
        self.conv7 = nn_sep.SeparableConv2d(in_channels=256, out_channels=256, kernel_size = 3, stride=1, padding="same")#, normalization_dw=None, normalization_pw=None), normalization_dw=None, normalization_pw=None)
        self.conv8 = nn_sep.SeparableConv2d(in_channels=256, out_channels=256, kernel_size = 3, stride=1, padding="same")#, normalization_dw=None, normalization_pw=None), normalization_dw=None, normalization_pw=None)

        self.conv9 = nn_sep.SeparableConv2d(in_channels=256, out_channels=512, kernel_size = 3, stride=2, padding="valid")#, normalization_dw=None, normalization_pw=None), normalization_dw=None, normalization_pw=None)

        self.conv10 = nn_sep.SeparableConv2d(in_channels=512, out_channels=512, kernel_size = 3, stride=1, padding="same")#, normalization_dw=None, normalization_pw=None), normalization_dw=None, normalization_pw=None)
        self.conv11 = nn_sep.SeparableConv2d(in_channels=512, out_channels=512, kernel_size = 3, stride=1, padding="same")#, normalization_dw=None, normalization_pw=None), normalization_dw=None, normalization_pw=None)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 10)
        self.act = nn.ReLU()


    def forward(self, x):
        x = self.act((self.conv1(x)))
        # print(x.size())
        x = self.act((self.conv2(x)))
        # print(x.size())
        x = self.act((self.conv3(x)))
        # print(x.size())
        x = self.act((self.conv4(x)))
        # print(x.size())
        x = self.act((self.conv5(x)))
        # print(x.size())
        x = self.act((self.conv6(x)))
        # print(x.size())
        x = self.act((self.conv7(x)))
        # print(x.size())
        x = self.act((self.conv8(x)))
        # print(x.size())
        x = self.act((self.conv9(x)))
        x = self.act((self.conv10(x)))
        x = self.act((self.conv11(x)))

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


class CIFAR8M(nn.Module):
    def __init__(self):
        super(CIFAR8M, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size = 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size = 3)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size = 3, stride=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size = 3)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size = 3)
        self.bn5 = nn.BatchNorm2d(128)

        self.conv6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size = 3, stride=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size = 3)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size = 3)
        self.bn8 = nn.BatchNorm2d(256)

        self.conv9 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size = 3, stride=1)
        self.bn9 = nn.BatchNorm2d(512)
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size = 3)
        self.bn10 = nn.BatchNorm2d(512)
        self.conv11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size = 3)
        self.bn11 = nn.BatchNorm2d(512)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 10)

        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        # print(x.size())
        x = self.act(self.bn2(self.conv2(x)))
        # print(x.size())
        x = self.act(self.bn3(self.conv3(x)))
        # print(x.size())
        x = self.act(self.bn4(self.conv4(x)))
        # print(x.size())
        x = self.act(self.bn5(self.conv5(x)))
        # print(x.size())
        x = self.act(self.bn6(self.conv6(x)))
        # print(x.size())
        x = self.act(self.bn7(self.conv7(x)))
        # print(x.size())
        x = self.act(self.bn8(self.conv8(x)))
        # print(x.size())
        x = self.act(self.bn9(self.conv9(x)))
        x = self.act(self.bn10(self.conv10(x)))
        x = self.act(self.bn11(self.conv11(x)))

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x
