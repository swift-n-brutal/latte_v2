# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 15:13:43 2016

@author: shiwu_001
"""

from .config import USE_GPU, DEVICE_ID, PYCAFFE_ROOT
#import sys
#sys.path.insert(0, PYCAFFE_ROOT)
if USE_GPU:
    from .util import get_block_shape, get_grid_shape, set_device, CAFFE_CUDA_CONTEXT
    print "Do not use caffe.set_mode_cpu()/set_mode_gpu()/set_device()"
    print "Use latte.set_device() instead"
    from . import math_func
from .solver import SGDSolver
from .softrelu_solver import SoftReLUSolver
from .newton_solver import NewtonSolver
from .solver_wgan import SolverWGAN
from .net import Net
from .blob import Blob
from .dataloader import CifarDataLoader, CifarTransformer, CifarDataLoaderMultiThreading
from .image_dataloader import ImageDataLoader, ImageDataLoaderPrefetch, ImageTransformer
from .rand_dataloader import RandDataLoader