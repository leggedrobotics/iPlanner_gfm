# ======================================================================
# Copyright (c) 2023 Fan Yang
# Robotic Systems Lab, ETH Zurich
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ======================================================================

import torch
from percept_net import PerceptNet
from vit import ViTFeatureExtractor, Dinov2FeatureExtractor
import torch.nn as nn


class PlannerNet(nn.Module):
    def __init__(self, encoder_channel=64, k=5):
        super().__init__()
        self.encoder = PerceptNet(layers=[2, 2, 2, 2])
        self.decoder = Decoder(512, encoder_channel, k)

    def forward(self, x, goal):
        x = self.encoder(x)
        x, c = self.decoder(x, goal)
        return x, c

class PlannerNetDino(nn.Module):
    def __init__(self, encoder_channel=64, k=5, encoder="dino", pretrained=False, pretrained_weights="models/teacher_checkpoint.pth", freeze_backbone=False):
        super().__init__()

        if encoder == "dinos16": # ViT-S16 (w/wo DINO Pretraining)
            self.encoder = ViTFeatureExtractor(freeze_backbone=freeze_backbone, pretrained=pretrained)
        
        elif encoder == "dinos16d": # Dino-S16-Depth
            self.encoder = Dinov2FeatureExtractor(freeze_backbone=freeze_backbone, pretrained_ckpt=pretrained_weights)
        else:
            raise ValueError("Invalid Encoder")

        self.skip_conv = nn.Sequential(nn.Conv2d(3, 128, kernel_size=16, stride=16), nn.BatchNorm2d(128), nn.ReLU())
        self.decoder = Decoder(512, encoder_channel, k)
        for name, param in self.named_parameters():
            print(f"Parameter: {name}, Requires Grad: {param.requires_grad}")


    def forward(self, x, goal):
        x1 = self.encoder(x)
        x2 = self.skip_conv(x)
        x = torch.cat((x1, x2), dim=1)
        x, c = self.decoder(x, goal)
        return x, c

class Decoder(nn.Module):
    def __init__(self, in_channels, goal_channels, k=5):
        super().__init__()
        self.k = k
        self.relu    = nn.ReLU(inplace=True)
        self.fg      = nn.Linear(3, goal_channels)
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d((in_channels + goal_channels), 512, kernel_size=5, stride=1, padding=1)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0);

        self.fc1   = nn.Linear(256 * 128, 1024) 
        self.fc2   = nn.Linear(1024, 512)
        self.fc3   = nn.Linear(512,  k*3)
        
        self.frc1 = nn.Linear(1024, 128)
        self.frc2 = nn.Linear(128, 1)

    def forward(self, x, goal):
        # compute goal encoding
        goal = self.fg(goal[:, 0:3])
        goal = goal[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
        # cat x with goal in channel dim
        x = torch.cat((x, goal), dim=1)
        # compute x
        x = self.relu(self.conv1(x))   # size = (N, 512, x.H/32, x.W/32)
        x = self.relu(self.conv2(x))   # size = (N, 512, x.H/60, x.W/60)
        x = torch.flatten(x, 1)

        f = self.relu(self.fc1(x))

        x = self.relu(self.fc2(f))
        x = self.fc3(x)
        x = x.reshape(-1, self.k, 3)

        c = self.relu(self.frc1(f))
        c = self.sigmoid(self.frc2(c))

        return x, c
