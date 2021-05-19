#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

@author: Hankui Peng

"""


import cv2
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
import utils.flow_transforms as flow_transforms


def ProduceAffMap(img_file, model):
    
    '''
    Given a trained model (.tar) and an image as input, 
    this function produces the affinity maps as output. 
    '''
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    input_transform = transforms.Compose([flow_transforms.ArrayToTensor()])

    img_ = cv2.imread(img_file)[:, :, ::-1].astype(np.float32) # may get 4 channel (alpha channel) for some format
    
    input_tensor = torch.tensor(img_.transpose(2,0,1)).to(device).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor) # 1 x 8 x H x W
    
    affinities = np.transpose(output.squeeze(0).cpu().numpy(), (1,2,0))
    
    return affinities 


def compute_edge_affinities_prob(edge, conn8 = 1):
    
    '''
    edge: Convert edge detection results to 8-channel format
    '''
        
    h,w = edge.shape
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    p2d = (1,1,1,1)
    
    edge_padded = F.pad(edge, p2d, mode='constant', value=-2) # the expanded tensor
    
    if conn8 == 1:
        aff_map = torch.zeros([1, 8, h, w])
        # top-left
        aff_map[:,0,:,:] = torch.max(edge_padded[1:-1,1:-1], edge_padded[:-2,:-2])        
        # top
        aff_map[:,1,:,:] = torch.max(edge_padded[1:-1,1:-1], edge_padded[:-2,1:-1]) # ground truth relationships         
        # top-right 
        aff_map[:,2,:,:] = torch.max(edge_padded[1:-1,1:-1], edge_padded[:-2,2:]) # ground truth relationships         
        # left 
        aff_map[:,3,:,:] = torch.max(edge_padded[1:-1,1:-1], edge_padded[1:-1,:-2]) # ground truth relationships         
        # right
        aff_map[:,4,:,:] = torch.max(edge_padded[1:-1,1:-1], edge_padded[1:-1,2:]) # ground truth relationships         
        # bottom-left
        aff_map[:,5,:,:] = torch.max(edge_padded[1:-1,1:-1], edge_padded[2:,:-2]) # ground truth relationships         
        # bottom
        aff_map[:,6,:,:] = torch.max(edge_padded[1:-1,1:-1], edge_padded[2:,1:-1]) # ground truth relationships         
        # bottom-right
        aff_map[:,7,:,:] = torch.max(edge_padded[1:-1,1:-1], edge_padded[2:,2:]) # ground truth relationships 
    else:
        aff_map = torch.zeros([1, 4, h, w])
        # top
        aff_map[:,0,:,:] = torch.max(edge_padded[1:-1,1:-1], edge_padded[:-2,1:-1]) # ground truth relationships         
        # right
        aff_map[:,1,:,:] = torch.max(edge_padded[1:-1,1:-1], edge_padded[1:-1,2:]) # ground truth relationships         
        # bottom
        aff_map[:,2,:,:] = torch.max(edge_padded[1:-1,1:-1], edge_padded[2:,1:-1]) # ground truth relationships         
        # left 
        aff_map[:,3,:,:] = torch.max(edge_padded[1:-1,1:-1], edge_padded[1:-1,:-2]) # ground truth relationships         
    
    return aff_map