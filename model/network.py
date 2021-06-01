#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

@author: Hankui Peng

"""

"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).
"""


import torch
import torch.nn as nn
import getopt
import math
import numpy
import os
import sys
import PIL
import PIL.Image


class Network(torch.nn.Module):
    
	def __init__(self):
        
		super(Network, self).__init__()

		self.netVggOne = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=8, out_channels=64, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False),
			torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False)
		)

		self.netVggTwo = torch.nn.Sequential(
			torch.nn.MaxPool2d(kernel_size=2, stride=2),
			torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False),
			torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False)
		)

		self.netVggThr = torch.nn.Sequential(
			torch.nn.MaxPool2d(kernel_size=2, stride=2),
			torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False),
			torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False),
			torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False)
		)

		self.netVggFou = torch.nn.Sequential(
			torch.nn.MaxPool2d(kernel_size=2, stride=2),
			torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False),
			torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False),
			torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False)
		)

		self.netVggFiv = torch.nn.Sequential(
			torch.nn.MaxPool2d(kernel_size=2, stride=2),
			torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False),
			torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False),
			torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False)
		)

		self.netScoreOne = torch.nn.Conv2d(in_channels=64, out_channels=8, kernel_size=1, stride=1, padding=0)
		self.netScoreTwo = torch.nn.Conv2d(in_channels=128, out_channels=8, kernel_size=1, stride=1, padding=0)
		self.netScoreThr = torch.nn.Conv2d(in_channels=256, out_channels=8, kernel_size=1, stride=1, padding=0)
		self.netScoreFou = torch.nn.Conv2d(in_channels=512, out_channels=8, kernel_size=1, stride=1, padding=0)
		self.netScoreFiv = torch.nn.Conv2d(in_channels=512, out_channels=8, kernel_size=1, stride=1, padding=0)

		self.netCombine = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=8*5, out_channels=8, kernel_size=1, stride=1, padding=0),
			torch.nn.Sigmoid()
		)

	def forward(self, tenInput):

		tenVggOne = self.netVggOne(tenInput)
		tenVggTwo = self.netVggTwo(tenVggOne)
		tenVggThr = self.netVggThr(tenVggTwo)
		tenVggFou = self.netVggFou(tenVggThr)
		tenVggFiv = self.netVggFiv(tenVggFou)
        
		tenScoreOne = self.netScoreOne(tenVggOne)
		tenScoreTwo = self.netScoreTwo(tenVggTwo)
		tenScoreThr = self.netScoreThr(tenVggThr)
		tenScoreFou = self.netScoreFou(tenVggFou)
		tenScoreFiv = self.netScoreFiv(tenVggFiv)
        
		tenScoreOne = torch.nn.functional.interpolate(input=tenScoreOne, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
		tenScoreTwo = torch.nn.functional.interpolate(input=tenScoreTwo, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
		tenScoreThr = torch.nn.functional.interpolate(input=tenScoreThr, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
		tenScoreFou = torch.nn.functional.interpolate(input=tenScoreFou, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
		tenScoreFiv = torch.nn.functional.interpolate(input=tenScoreFiv, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        
		return self.netCombine(torch.cat([ tenScoreOne, tenScoreTwo, tenScoreThr, tenScoreFou, tenScoreFiv ], 1))


class MyResBlock(nn.Module):
    
    expansion = 1

    def __init__(self, in_channels, out_channels):
        super(MyResBlock, self).__init__()
        self.pad1 = nn.ReplicationPad2d(1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=True)
        self.in1 = nn.InstanceNorm2d(out_channels, affine=False, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.pad2 = nn.ReplicationPad2d(1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=True)
        self.in2 = nn.InstanceNorm2d(out_channels, affine=False, track_running_stats=True)

    def forward(self, x):
        
        residual = x
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.in1(out)
        out = self.relu(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.in2(out)
        out += residual
        out = self.relu(out)

        return out


class DAL(nn.Module):
    
    def __init__(self, nr_channel, conv1_size):
    
        super(DAL, self).__init__()
        
        pad_size = int((conv1_size - 1) / 2)
        self.pad1 = nn.ReplicationPad2d((pad_size, pad_size, pad_size, pad_size))     # left, right, top, bottom
        
        self.conv1 = nn.Conv2d(3, nr_channel, kernel_size=(conv1_size, conv1_size), stride=1, padding=0, bias=True)    
        self.in1 = nn.InstanceNorm2d(nr_channel, affine=False, track_running_stats=True)    
        self.relu1 = nn.ReLU()
    
        self.res2 = MyResBlock(nr_channel, nr_channel)
        self.res3 = MyResBlock(nr_channel, nr_channel)
        self.res4 = MyResBlock(nr_channel, nr_channel)
        self.hednet = Network()
        
        self.conv5 = nn.Conv2d(nr_channel, 8, kernel_size=1, stride=1, padding=0, bias=True)
        self.in5 = nn.InstanceNorm2d(8, affine=False, track_running_stats=True)
        self.sigm5 = nn.Sigmoid()
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        
        image = self.pad1(x)
        out = self.conv1(image)
        out = self.in1(out)
        out = self.relu1(out)
       
        # Residual blocks 
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)
       
        # HED architecture 
        out = self.hednet(out)
        
        return out