# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 14:12:46 2017

@author: shiwu_001
"""

from config import DTYPE, USE_GPU
import pycuda.gpuarray as garr
from numpy import zeros

class Blob(object):
    def __init__(self, blob, copy=False):
        self.hold_ = copy
        self.blob_ = None
        self.count = blob.count
        self.shape = blob.shape
        if copy:
            # only used when the blob holds data by itself
            # to save unnecessary copys
            self.last_data_cpu_ = True
            self.last_diff_cpu_ = True
            # lazy scheme to allocate memory
#            self.data_ = zeros(blob.shape, dtype=DTYPE)
#            self.diff_ = zeros(blob.shape, dtype=DTYPE)
            self.data_ = None
            self.diff_ = None
            if USE_GPU:
#                self.gpu_data_ = garr.zeros(shape=blob.shape, dtype=DTYPE)
#                self.gpu_diff_ = garr.zeros(shape=blob.shape, dtype=DTYPE)
                self.gpu_data_ = None
                self.gpu_diff_ = None
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
            if self.data_ is None:
                self.data_ = zeros(self.shape, dtype=DTYPE)
            if not self.last_data_cpu_:
                self.gpu_data_.get(self.data_)
                self.last_data_cpu_ = True
        else:
            self.data_ = self.blob_.data
        return self.data_
    
    @property
    def gpu_data(self):
        if self.hold_:
            if self.gpu_data_ is None:
                self.gpu_data_ = garr.zeros(shape=self.shape, dtype=DTYPE)
            if self.last_data_cpu_ and self.data_ is not None:
                self.gpu_data_.set(self.data_)
            self.last_data_cpu_ = False
        else:
            # call gpu_data_ptr to update data on the device
            self.blob_.gpu_data_ptr
        return self.gpu_data_

    @property
    def diff(self):
        if self.hold_:
            if self.diff_ is None:
                self.diff_ = zeros(self.shape, dtype=DTYPE)
            if not self.last_diff_cpu_:
                self.gpu_diff_.get(self.diff_)
                self.last_diff_cpu_ = True
        else:
            self.diff_ = self.blob_.diff
        return self.diff_
        
    @property
    def gpu_diff(self):
        if self.hold_:
            if self.gpu_diff_ is None:
                self.gpu_diff_ = garr.zeros(shape=self.shape, dtype=DTYPE)
            if self.last_diff_cpu_ and self.diff_ is not None:
                self.gpu_diff_.set(self.diff_)
            self.last_diff_cpu_ = False
        else:
            # call gpu_diff_ptr to update diff on the device
            self.blob_.gpu_diff_ptr
        return self.gpu_diff_
        
    def share_data(self, other):
        if not (self.hold_ or other.hold_):
            self.blob_.share_data(other.blob_)
            self.data_ = None
            self.gpu_data_ = garr.GPUArray(shape=self.shape, dtype=DTYPE,
                                           gpudata=self.blob_.gpu_data_ptr)
        else:
            raise ValueError("Can't share data from or to a copied blob.")
    
    def share_diff(self, other):
        if not (self.hold_ or other.hold_):
            self.blob_.share_diff(other.blob_)
            self.diff_ = None
            self.gpu_diff_ = garr.GPUArray(shape=self.shape, dtype=DTYPE,
                                           gpudata=self.blob_.gpu_diff_ptr)
        else:
            raise ValueError("Can't share diff from or to a copied blob.")