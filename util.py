# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 16:36:16 2016

@author: shiwu_001
"""

import pycuda.driver as cuda
import caffe

def get_block_shape():
    """
        Return number of threads per block
    """
    return (caffe.cuda_num_threads(), 1, 1)

def get_grid_shape(count):
    """
        Compute number of blocks per grid
    """
    return (caffe.get_blocks(count), 1)

def set_device(device_id):
    global CAFFE_CUDA_CONTEXT
    CAFFE_CUDA_CONTEXT.set_device(device_id)
    
class CaffeCudaContext:
    """
        Share CUDA Context from Caffe.
        NOTE: PyCuda uses 'cuCtxAttach' and 'cuCtxDetach' functions to get the
            current CUDA context (and increase/decrease the use counter) in
            Caffe, if Caffe is initialized before PyCuda. These functions are
            deprecated. 'cuCtxGetCurrent' is recommended after CUDA (v6.5).
    """
    def __init__(self):
        if not caffe.check_mode_gpu():
            raise ValueError("Unable to init PyCuda. Caffe mode != GPU.")
        # Attach to Caffe's context and increase use counter
        self.ctx_ = cuda.Context.attach()
        
    def set_device(self, device_id):
        if device_id != caffe.get_device():
            self.ctx_.detach()
            caffe.set_device(device_id)
            self.ctx_ = cuda.Context.attach()
        
    def __del__(self):
        # Detach the context and decrease use counter
        self.ctx_.detach()

caffe.set_mode_gpu()
CAFFE_CUDA_CONTEXT = CaffeCudaContext()