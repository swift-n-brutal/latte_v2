# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 19:29:04 2016

@author: shiwu_001
"""

from .config import DTYPE
from .blob import Blob
from .solver import SGDSolver
from .math_func import softlinear, softsigmoid, eltmul
import time

class SoftReLUSolver(SGDSolver):
    def __init__(self, *args, **kwargs):
        super(SoftReLUSolver, self).__init__(*args, **kwargs)
        self.relu_layer_indices, self.bottoms, self.tops, self.back_mults = \
            self._get_relu_layer_indices_and_blobs()
        self.soft_a = 0.
        self.soft_type = 'linear'
    
    def _get_relu_layer_indices_and_blobs(self):
        net = self.net
        relu_layer_indices = list()
        bottoms = list()
        tops = list()
        back_mults = list()
        for i, l in enumerate(self.net.layers):
            if l.type == "ReLU":
                # layer indices
                relu_layer_indices.append(i)
                # bottoms
                blob_ids = net._bottom_ids(i)
                assert(len(blob_ids) == 1)
                blob_name = net._blob_names[blob_ids[0]]
                bottoms.append(Blob(net.blobs[blob_name], copy=False))
                # tops
                blob_ids = net._top_ids(i)
                assert(len(blob_ids) == 1)
                blob_name = net._blob_names[blob_ids[0]]
                tops.append(Blob(net.blobs[blob_name], copy=False))
                # multipliers in the backward
                back_mults.append(Blob(net.blobs[blob_name], copy=True))
        return relu_layer_indices, bottoms, tops, back_mults
            
    def _net_forward_gpu(self):
        net = self.net
        soft_a = DTYPE(self.soft_a)
        if self.soft_type == 'linear':
            soft_func = softlinear
        elif self.soft_type == 'sigmoid':
            soft_func = softsigmoid
        else:
            raise ValueError("Invalid soft type: %s" % self.soft_type)
        last_lname = None
        for rlid, bt, bm in zip(self.relu_layer_indices, self.bottoms, self.back_mults):
            net.forward(start=last_lname, end=net._layer_names[rlid-1])
            soft_func(soft_a, bt.gpu_data, bm.gpu_data)
            last_lname = net._layer_names[rlid]
        # forward the last few layers
        return net.forward(start=last_lname, end=None)
        
    def _net_backward_gpu(self):
        net = self.net
        last_lname = None
        for rlid, bt, tp, bm in zip(reversed(self.relu_layer_indices), reversed(self.bottoms), reversed(self.tops), reversed(self.back_mults)):
            net.backward(start=last_lname, end=net._layer_names[rlid+1])
            # apply soft backward to ReLU
            blob_name = net._blob_names[net._bottom_ids(rlid+1)[0]]
            net.blobs[blob_name].gpu_diff_ptr
            eltmul(tp.gpu_diff, bm.gpu_data, bt.gpu_diff)
            # skip the backward of ReLU
            last_lname = net._layer_names[rlid-1]
#            last_lname = net._layer_names[rlid]
        # backward the last few layers
        return net.backward(start=last_lname, end=None)
    
    def _net_forward(self):
        return self._net_forward_gpu()
        
    def _net_backward(self):
        return self._net_backward_gpu()
    
    def set_soft_a(self, soft_a):
        self.soft_a = soft_a
        
    def set_soft_type(self, soft_type):
        self.soft_type = soft_type
        
    def step(self, lr, mom, decay, dist_type, verbose=False):
        batchid_ = None
#        stime = time.time()
        self._clear_diff(self.net)
#        etime = time.time()
#        print "clrd %.3f|" % ((etime - stime)*100),
#        stime = etime
        self.net.load_data(batchid_)
#        etime = time.time()
#        print "data %.3f|" % ((etime - stime)*100),
#        stime = etime
        output_fw_ = self._net_forward()
#        etime = time.time()
#        print "forw %.3f|" % ((etime - stime)*100),
#        stime = etime
        output_bw_ = self._net_backward()
#        etime = time.time()
#        print "bckw %.3f|" % ((etime - stime)*100),
#        stime = etime
        self.update(self.net, lr, mom, decay, dist_type)
#        etime = time.time()
#        print "updt %.3f|" % ((etime - stime)*100)
        return output_fw_, output_bw_, batchid_