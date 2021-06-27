#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import cv2
import os


class SELayer(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SELayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = nn.functional.avg_pool2d(x, (h, w)).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class Conv_bn_relu(nn.Module):
    def __init__(self, inp, oup, kernel_size, stride, pad):
        super(Conv_bn_relu, self).__init__()
        self.conv = nn.Conv2d(inp, oup, kernel_size, stride, pad, bias=False)
        self.bn = nn.BatchNorm2d(oup, momentum=0.05)
        self.relu = nn.ReLU()
        self.se = SELayer(oup)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.se(out)
        return out


class Conv(nn.Module):
    def __init__(self, inp, oup, kernel_size, stride, pad):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(inp, oup, kernel_size, stride, pad, bias=False)

    def forward(self, x):
        out = self.conv(x)
        return out


class ResiBlock(nn.Module):
    def __init__(self, inp):
        super(ResiBlock, self).__init__()

        self.conv1 = nn.Conv2d(inp, inp, 3, 1, 1, groups=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inp, momentum=0.05)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(inp, inp, 3, 1, 1, groups=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inp, momentum=0.05)

        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(inp)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        return self.relu(x + out)


class MTPoseNet(nn.Module):
    def __init__(self, args):
        super(MTPoseNet, self).__init__()
        self.in_channel = args['inplanes']
        self.out_channel = args['heatmaps_num']
        self.points_end_index = args['points_num'] - args['heatmaps_num']
        if self.points_end_index == 0:
            self.points_end_index = self.out_channel

        self.finetune_path = args['finetune_path']

        self.stage_1 = nn.Sequential(
            Conv_bn_relu(self.in_channel, 16, 3, 2, 1),
            Conv_bn_relu(16, 16, 3, 2, 1),
            ResiBlock(16),
        )

        self.stage_1_1 = nn.Sequential(
            Conv_bn_relu(16, 32, 3, 2, 1),
            ResiBlock(32),
            ResiBlock(32),
            ResiBlock(32),
            Conv_bn_relu(32, 32, 3, 1, 1),
        )

        self.stage_2 = nn.Sequential(
            Conv_bn_relu(32, 48, 3, 2, 1),
            ResiBlock(48),
            ResiBlock(48),
            ResiBlock(48),
            ResiBlock(48),
            ResiBlock(48),
            ResiBlock(48),
        )

        self.stage_2_combine_2 = nn.Sequential(
            Conv(48, 32, 1, 1, 0),
            nn.Upsample(scale_factor=2),
        )

        self.stage_1_combine_1 = Conv(16, 16, 1, 1, 0)
        self.stage_1_combine_2 = nn.Sequential(
            Conv(32, 16, 1, 1, 0),
            nn.Upsample(scale_factor=2)
        )

        self.stage_3 = nn.Sequential(
            ResiBlock(32),
        )

        self.stage_4 = nn.Sequential(
            ResiBlock(16),
            Conv_bn_relu(16, 96, 1, 1, 0)
        )

        self.conv_out = Conv(96, self.out_channel, 1, 1, 0)

        self.conv1 = Conv(96, 32, 1, 1, 0)
        self.conv2 = Conv(48, 32, 1, 1, 0)
        self.fc = nn.Linear(129024, 126)

        self.apply(self._init_weights)

        # initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        if self.finetune_path is not None:
            checkpoint = torch.load(self.finetune_path, map_location=lambda storage, loc: storage)
            checkpoint_ = {}

            for k, v in checkpoint.items():
                checkpoint_[k] = v

            checkpoint = checkpoint_
            model_dict = self.state_dict()
            checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict}
            model_dict.update(checkpoint)

            self.load_state_dict(model_dict)

    def forward(self, x, target=None):
        # with torch.no_grad:
        out_1 = self.stage_1(x)

        out_1_1 = self.stage_1_1(out_1)

        out_2 = self.stage_2(out_1_1)

        out_2_sum = out_1_1 + self.stage_2_combine_2(out_2)

        out_3 = self.stage_3(out_2_sum)

        out_3_sum = self.stage_1_combine_1(out_1) + self.stage_1_combine_2(out_3)

        out_4 = self.stage_4(out_3_sum)

        out = self.conv_out(out_4)

        feat1 = self.conv1(out_4)
        feat2 = self.conv2(out_2)

        reg = torch.cat((feat1.view(feat1.size(0), -1), feat2.view(feat2.size(0), -1)), 1)
        reg_out = torch.cat((out_3.view(out_3.size(0), -1), reg), 1)
        out = self.fc(reg_out)

        return out, out_1_1, out_2_sum, out_3_sum, out_4

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                init.constant_(m.bias, 0.)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1.)
            init.constant_(m.bias, 0.)
        elif isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0.)


def get_model():
    model_path = '/root/group-incubation-bj/contour/base_cnn/snapshot_best_G_model'
    net_args = {
        'inplanes': 3,
        'heatmaps_num': 63,
        'points_num': 63,
        'finetune_path': None}
    target_net = MTPoseNet(net_args)

    # load param
    model_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    param1 = target_net.state_dict()
    for key, value in model_dict.items():
        # key1 = key.split('.', 1)[1]
        key1 = key
        if key1 in param1.keys() and key1 != 'conv_out.conv.weight':
            param1[key1] = value
        else:
            print("not match", key1)
    target_net.load_state_dict(param1)
    return target_net


if __name__ == '__main__':
    input = torch.ones(10, 3, 256, 128)
    model = get_model()
    out = model(input)
