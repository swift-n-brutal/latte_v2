# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 15:35:55 2017

@author: shiwu_001
"""

from .blob import Blob

class Layer(object):
    def __init__(self, caffe_layer):
        self.type = caffe_layer.type
        self.blobs = list()
        for b in caffe_layer.blobs:
            self.blobs.append(Blob(b))
    
