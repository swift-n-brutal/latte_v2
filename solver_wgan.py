# -*- coding: utf-8 -*-
"""
Created on Fri Mar 03 10:42:33 2017

@author: shiwu_001
"""

import time

from blob import Blob
from math_func import sqrx, sqrtx, axpby, xdivyeps, axpb, clipab, setx, sumsqr

import numpy as np
DTYPE = np.float32

class SolverWGAN(object):
    PARAM_LAYER_TYPE = ['Convolution', 'Scale', 'InnerProduct']
    def __init__(self, net_g, net_d, test_net_g=None, test_net_d=None,
                 lr=0.00005, mom=0.9, mom2=0.99, eps=1e-8, c=0.01,
                 inner_iter=5, solver_type='RMSProp'):
        self.net_g = net_g
        self.net_d = net_d
        self.test_net_g = test_net_g
        self.test_net_d = test_net_d
        self.param_g = self._init_param(net_g)
        self.param_d = self._init_param(net_d)
        self.history_g = self._init_history(net_g)
        self.history_d = self._init_history(net_d)
        self.lr = lr
        self.mom = mom
        self.mom2 = mom2
        self.accum_mult = 0.0
        self.accum_mult2 = 0.0
        self.eps = eps
        self.c = c
        self.inner_iter = inner_iter
        self.gen_name = self.net_g.outputs[0]
        self.image_name = self.net_d.inputs[0]
        self.score_name = self.net_d.outputs[0]
        self._gen_blob = Blob(self.net_g.blobs[self.gen_name], copy=False)
        self._image_blob = Blob(self.net_d.blobs[self.image_name], copy=False)
        self._score_blob = Blob(self.net_d.blobs[self.score_name], copy=False)
        
    def _init_history(self, net):
        history = list()
        for l in net.layers:
            blobs = None
            if l.type in self.PARAM_LAYER_TYPE:
                blobs = list()
                for b in l.blobs:
                    blobs.append(Blob(b, copy=True))
            history.append(blobs)
        return history
    
    def _init_param(self, net):
        param = list()
        for l in net.layers:
            blobs = None
            if l.type in self.PARAM_LAYER_TYPE:
                blobs = list()
                for b in l.blobs:
                    blobs.append(Blob(b, copy=False))
            param.append(blobs)
        return param

    def _clear_diff(self, param):
        for p in param:
            if p is not None:
                for b in p:
                    b.gpu_diff.fill(DTYPE(0))
    
    def _clip_param_gpu(self, param):
        for p in param:
            if p is not None:
                for bp in p:
                    clipab(bp.gpu_data, self.c, -self.c, bp.gpu_data)
                    
    def _clip_param(self, param):
        self._clip_param_gpu(param)
    
    def _update_rmsprop_gpu(self, param, history):
        self.accum_mult2 = self.accum_mult2 * self.mom2 + 1.0
        for (p, h) in zip(param, history):
            if p is not None:
                for (bp, bh) in zip(p, h):
                    # history.diff = param.diff ^ 2
                    sqrx(bp.gpu_diff, bh.gpu_diff)
                    # history.data = history.data * mom + history.diff
                    axpby(DTYPE(self.mom2), bh.gpu_data, DTYPE(1.), bh.gpu_diff, bh.gpu_data)
                    # history.diff = history.data / accum_mult
                    axpb(DTYPE(1./self.accum_mult2), bh.gpu_data, DTYPE(0.), bh.gpu_diff)
                    # history.diff = sqrt( history.diff )
                    sqrtx(bh.gpu_diff, bh.gpu_diff)
                    # param.diff = param.diff / (history.diff + eps)
                    xdivyeps(bp.gpu_diff, bh.gpu_diff, DTYPE(self.eps), bp.gpu_diff)
                    # param.data = param.data - lr * param.diff
                    axpby(DTYPE(1), bp.gpu_data, DTYPE(-self.lr), bp.gpu_diff, bp.gpu_data)
                    
    def _update_rmsprop(self, param, history):
        self._update_rmsprop_gpu(param, history)
    
    def _inner_step_d(self):
        loss_d = []
        for itr in range(self.inner_iter):
            loss_d_i = []
            self._clear_diff(self.param_d)
            # real data for discriminator
            self.net_d.load_data(None)
            self.net_d.forward()
            loss_d_i.append(self._score_blob.gpu_data.get())
            self._score_blob.gpu_diff.fill(DTYPE(-1.0))
            self.net_d.backward()
            # generated data for discriminator
            self.net_g.load_data(None)
            self.net_g.forward()
            setx(self._gen_blob.gpu_data, self._image_blob.gpu_data)
            self.net_d.forward()
            loss_d_i.append(self._score_blob.gpu_data.get())
            self._score_blob.gpu_diff.fill(DTYPE(1.0))
            self.net_d.backward()
            # update discriminator
            self._update_rmsprop(self.param_d, self.history_d)
            # clip parameters of discriminator
            self._clip_param(self.param_d)
            loss_d.append(loss_d_i)
        return loss_d
    
    def _inner_step_g(self):
        self._clear_diff(self.param_g)
        # forward
        self.net_g.load_data(None)
        self.net_g.forward()
        setx(self._gen_blob.gpu_data, self._image_blob.gpu_data) 
        self.net_d.forward()
        # loss
        loss_g = self._score_blob.gpu_data.get()
        # backward
        self._score_blob.gpu_diff.fill(DTYPE(-1.0))
        self.net_d.backward()
        setx(self._image_blob.gpu_diff, self._gen_blob.gpu_diff)
#        print sumsqr(self._gen_blob.gpu_diff).get()
        self.net_g.backward()
        # update
        self._update_rmsprop(self.param_g, self.history_g)
        return loss_g
            
    def step(self, solver_type='RMSProp', verbose=False):
        # step for discriminator
        loss_d = self._inner_step_d()
        # step for generator
        loss_g = self._inner_step_g()
        return [loss_d, loss_g]