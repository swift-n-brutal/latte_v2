# -*- coding: utf-8 -*-
"""
Created on Fri Mar 03 18:31:58 2017

@author: shiwu_001
"""

import os.path as osp
import sys
#import google.protobuf as pb
#from argparse import ArgumentParser

CAFFE_ROOT = r'E:\projects\cpp\caffe-windows-ms'
PYCAFFE_PATH = osp.join(CAFFE_ROOT, r'Build\x64\Release\pycaffe')
if PYCAFFE_PATH not in sys.path:
    sys.path.insert(0, PYCAFFE_PATH)
    
from caffe import TRAIN
from caffe import Net as CaffeNet

class Net(CaffeNet):
    def __init__(self, net_def, phase=TRAIN, net_param=None):
        if net_param is None:
            super(Net, self).__init__(net_def, phase)
        else:
            super(Net, self).__init__(net_def, net_param, phase)
        self.dataloader = None
        self.data_blob_names = None
        self.data_blobs = None
    
    def set_dataloader(self, dataloader):
        self.dataloader = dataloader
    
    def set_data_blobs(self, blob_names):
        self.data_blob_names = blob_names
        self.data_blobs = [self.blobs[name] for name in self.data_blob_names]
    
    def load_data(self, batchids=None):
        if self.dataloader is not None:
            self.dataloader.fill_input(self.data_blobs, self.data_blob_names,
                                       batchids=batchids)