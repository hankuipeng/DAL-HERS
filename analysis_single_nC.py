#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

@author: Hankui Peng

"""


## import necessary modules 
# system 
from skimage.segmentation import mark_boundaries
from skimage import io
import skimage
from glob import glob 
import numpy as np
import argparse 
import random 
import torch
import time 
import sys 
import cv2
import os 

# local
sys.path.insert(0, "../pybuild")
sys.path.insert(0, "pybuild")
from utils.analysis_util import *
from utils.hed_edges import *
from model.network import *
import hers_superpixel


## input arguments 
parser = argparse.ArgumentParser(description='Hierarchical Entropy Rate Superpixel (HERS) Segmentation on a folder of images',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--nC', default=200, type=int, help='the number of desired superpixels')
parser.add_argument('--pretrained', default='./pretrained/DGSS_loss=bce-rgb_date=23Feb2021.tar', help='path to the pretrained model')
parser.add_argument('--input_dir', default='./sample_imgs/input/', help='path to images folder')
parser.add_argument('--output_dir', default='./sample_imgs/output/', help='path to output folder')
parser.add_argument('--output_suff', default='', help='suffix to the output file')
parser.add_argument('--edge', default=True, help='whether to incorporate edge information')
parser.add_argument('--device', default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), help='default device (CPU / GPU)')
args = parser.parse_args()

random.seed(100)


## main function 
def main():
    
    data_type = np.float32
    
    # read all the image files in the folder 
    tst_lst = glob(args.input_dir + '*.jpg')
    tst_lst.sort()

    # load the model 
    network_data = torch.load(args.pretrained, map_location=args.device)
    model = DGSS(nr_channel=8, conv1_size=7)
    model.load_state_dict(network_data['state_dict'])
    model.eval()
    
    # for each image:
    for n in range(len(tst_lst)):
            
        ## input image 
        img_file = tst_lst[n]
        imgId = os.path.basename(img_file)[:-4]
        image = cv2.imread(img_file)
        input_img = image.astype(data_type)
        h, w, ch = image.shape
        
        
        ## input affinities 
        affinities = ProduceAffMap(img_file, model)
        input_affinities = affinities 
         
        
        ## HED edge information 
        if args.edge:
            Input = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(PIL.Image.open(img_file))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32)))*(1.0 / 255.0)
            edge_prob = estimate(Input) 
            input_edge = np.array(edge_prob.squeeze(0), dtype=data_type)
        else:
            input_edge = np.ones((h, w), dtype=data_type) # Provide no external edge information by default 

        
        ## build the hierarchical segmentation tree 
        start = time.time()
        bosupix = hers_superpixel.BoruvkaSuperpixel()
        bosupix.build_2d(input_img, input_affinities, input_edge)
        end = time.time()
        
        
        ## segmentation with user-defined number of superpixels
        sp_label = bosupix.label(args.nC)
        output_img = np.max(image) * mark_boundaries(image, sp_label.astype(int), color = (0,0,255)) # candidate color choice: (220,20,60)
        
        # output the label map as a csv file 
        save_csv_path = args.output_dir + str(args.nC) + '/csv/'
        if not os.path.isdir(save_csv_path):
            os.makedirs(save_csv_path)    
        label_map_path = save_csv_path + imgId + args.output_suff + '.csv'
        np.savetxt(label_map_path, sp_label.astype(int), fmt='%i', delimiter=",") 
        
        # output the visualisation    
        save_png_path = args.output_dir + str(args.nC) + '/png/'
        if not os.path.isdir(save_png_path):
            os.makedirs(save_png_path)    
        spixl_save_name = save_png_path + imgId + args.output_suff + '.png'
        cv2.imwrite(spixl_save_name, output_img)
        
        # save the run times 
        elapsed_time = (end - start)*1000
        
        print("Just finished the {0}th image for nC={1}, with HERS run time of {2:.2f} ms".format(int(n), args.nC, elapsed_time))
        
            
if __name__ == '__main__':
    
    main()