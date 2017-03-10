# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 15:15:56 2016

@author: shiwu_001
"""

import os.path as osp

CAFFE_ROOT = '../..'
CAFFE_INCLUDE = osp.join(CAFFE_ROOT, 'include')
LATTE_ROOT = osp.join(CAFFE_ROOT, 'examples')
PYCAFFE_ROOT = osp.join(CAFFE_ROOT, 'Build', 'x64', 'Release', 'pycaffe')

USE_GPU = True
DEVICE_ID = 1

import numpy
DTYPE = numpy.float32