# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 14:12:46 2017

@author: shiwu_001
"""

from .config import DTYPE, USE_GPU
import pycuda.gpuarray as garr
from numpy import zeros

class Blob(object):
    def __init__(self, blob, copy=False):
        self.hold_ = copy
        if copy:
            # only used when the blob holds data by itself
            # to save unnecessary copys
            self.last_data_cpu_ = True
            self.last_diff_cpu_ = True
            self.data_ = zeros(blob.shape, dtype=DTYPE)
            self.diff_ = zeros(blob.shape, dtype=DTYPE)
            if USE_GPU:
                self.gpu_data_ = garr.zeros(shape=blob.shape, dtype=DTYPE)
                self.gpu_diff_ = garr.zeros(shape=blob.shape, dtype=DTYPE)
        else:
            self.blob_ = blob
            self.data_ = None
            self.diff_ = None
            if USE_GPU:
                self.gpu_data_ = garr.GPUArray(shape=blob.shape, dtype=DTYPE,
                                               gpudata=blob.gpu_data_ptr)
                self.gpu_diff_ = garr.GPUArray(shape=blob.shape, dtype=DTYPE,
                                               gpudata=blob.gpu_diff_ptr)
    
    @property
    def data(self):
        if self.hold_:
            if not self.last_data_cpu_:
                self.gpu_data_.get(self.data_)
                self.last_data_cpu_ = True
        else:
            self.data_ = self.blob_.data
        return self.data_
    
    @property
    def gpu_data(self):
        if self.hold_:
            if self.last_data_cpu_:
                self.gpu_data_.set(self.data_)
                self.last_data_cpu_ = False
        else:
            # call gpu_data_ptr to update data on the device
            self.blob_.gpu_data_ptr
        return self.gpu_data_

    @property
    def diff(self):
        if self.hold_:
            if not self.last_diff_cpu_:
                self.gpu_diff_.get(self.diff_)
                self.last_diff_cpu_ = True
        else:
            self.diff_ = self.blob_.diff
        return self.diff_
        
    @property
    def gpu_diff(self):
        if self.hold_:
            if self.last_diff_cpu_:
                self.gpu_diff_.set(self.diff_)
                self.last_diff_cpu_ = False
        else:
            # call gpu_diff_ptr to update diff on the device
            self.blob_.gpu_diff_ptr
        return self.gpu_diff_