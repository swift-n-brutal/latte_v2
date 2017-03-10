# -*- coding: utf-8 -*-
"""
Created on Fri Mar 03 16:38:38 2017

@author: shiwu_001
"""

import numpy as np

class RandDataLoader(object):
    def __init__(self, rand_type="gaussian", std=1.0, seed=None):
#        super(RandDataLoader, self).__init__(seed=seed)
        self.rand = np.random.RandomState(seed)
        self.rand_type = rand_type
        self.std = std
    
    def fill_input(self, blobs, blob_names, batchids):
        if self.rand_type == "gaussian":
            for b in blobs:
                b.data[...] = self.std * self.rand.randn(b.count).reshape(b.shape)
        elif self.rand_type == "uniform":
            a = np.sqrt(3) * self.std
            for b in blobs:
                b.data[...] = self.rand.uniform(-a, a, size=b.shape)
        else:
            raise ValueError("Not supported rand_type: %s" %
                             self.rand_type)
