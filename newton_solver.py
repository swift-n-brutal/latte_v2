# -*- coding: utf-8 -*-
"""
Created on Thu Dec 01 21:14:13 2016

@author: shiwu_001
"""

from .config import DTYPE
#from .blob import Blob
from .solver import SGDSolver
from .math_func import sumsqr, axpb
import numpy as np
import time

class NewtonSolver(SGDSolver):
    def __init__(self, *args, **kwargs):
        super(NewtonSolver, self).__init__(*args, **kwargs)
        self.loss_blob = 'loss'
        self.loss = DTYPE(0)
        self.EPSILON = 1e-1
        self.norm_type = 'All' # 'All' or 'Ind'
    
    def _net_forward_gpu(self):
        net = self.net
        ofw = net.forward()
        self.loss = np.copy(ofw[self.loss_blob])
        return ofw
        
    def _net_backward_gpu(self):
        net = self.net
        obw = net.backward()
        net_blobs = self.net_blobs
        loss = self.loss
        if self.norm_type == 'All':
            ssqr = 0.
            for i, l in enumerate(net.layers):
                if l.type in ['Convolution', 'Scale', 'InnerProduct']:
                    for blb in net_blobs[i]:
                        ssqr += sumsqr(blb.gpu_diff).get()
            a = loss / (ssqr + self.EPSILON)
            for i, l in enumerate(net.layers):
                if l.type in ['Convolution', 'Scale', 'InnerProduct']:
                    for blb in net_blobs[i]:
                        axpb(DTYPE(a), blb.gpu_diff, DTYPE(0), blb.gpu_diff)
#            print "(%f, %.3f)" % (ssqr, a),
            obw['a'] = ssqr
        elif self.norm_type == 'Ind':
            count = 0
            s = 0.
            for i, l in enumerate(net.layers):
                if l.type in ['Convolution', 'Scale', 'InnerProduct']:
                    for blb in net_blobs[i]:
                        ssqr = sumsqr(blb.gpu_diff).get()
                        a = loss / (ssqr + self.EPSILON)
                        count += 1
                        s += a
                        axpb(DTYPE(a), blb.gpu_diff, DTYPE(0), blb.gpu_diff)
            obw['a'] = s / count
        else:
            raise ValueError('Invalid norm_type')
#        print "%.3f" % (s / count), 
        return obw
        
    def _net_forward(self):
        return self._net_forward_gpu()
    
    def _net_backward(self):
        return self._net_backward_gpu()
    
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