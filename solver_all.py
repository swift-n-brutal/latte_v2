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

class Solver(object):
    PARAM_LAYER_TYPE = ['Deconvolution', 'Convolution', 'Scale', 'InnerProduct']
    HISTORY_SOLVER_TYPE = ['SGD', 'RMSProp', 'Adam']
    HISTORY2_SOLVER_TYPE = ['Adam'] 
    def __init__(self, nets, test_nets=None, lr=0.00005, decay=0.,
                 mom=0.9, mom2=0.99, eps=1e-8, solver_type='RMSProp'):
        self.nets = nets
        self.test_nets = test_nets
        self.params = list()
        self.historys = list()
        self.history2s = list()
        for net in self.nets:
            # init blobs for parameter
            self.params.append(self._init_param(net_g))
            # init blobs for history
            if solver_type in Solver.HISTORY_SOLVER_TYPE:
                self.historys.append(self._init_history())
            else:
                self.historys.append(None)
            if solver_type in Solver.HISTORY2_SOLVER_TYPE:
                self.history2s.append(self._init_history())
            else:
                self.history2s.append(None)
        self.lr = lr
        self.decay = decay
        self.mom = mom
        self.mom2 = mom2
        self.accum_mult = 0.0
        self.accum_mult2 = 0.0
        self.eps = eps
        self.solver_type = solver_type
        
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
    
    def _updata_sgd_gpu(self, param, history):
        self.accum_mult = self.accum_mult * self.mom + 1.0
        for (p, h) in zip(param, history):
            if p is not None:
                for (bp, bh) in zip(p, h):
                    # param.diff = param.diff + param.data * decay
                    if self.decay != 0:
                        axpby(DTYPE(1.), bp.gpu_diff, DTYPE(self.decay), bp.gpu_data, bp.gpu_diff)
                    # history.data = history.data * mom + param.diff
                    axpby(DTYPE(self.mom), bh.gpu_data, DTYPE(1.), bp.gpu_diff, bh.gpu_data)
                    # param.diff = history.data / accum_mult
                    axpb(DTYPE(1./self.accum_mult), bh.gpu_data, DTYPE(0.), bp.gpu_diff)
                    # param.data = param.data - lr * param.diff
                    axpby(DTYPE(1.), bp.gpu_data, DTYPE(-self.lr), bp.gpu_diff, bp.gpu_data)
    
    def _update_sgd(self, param, history):
        self._update_sgd_gpu(param, history)

    def _update(self, param, history, history2=None):
	if self.solver_type == 'SGD':
            self._update_sgd(param, history)
        elif self.solver_type == 'RMSProp':
            self._update_rmsprop(param, history)
        elif self.solver_type == 'Adam':
            raise ValueError('To be implemented solver type %s' % self.solver_type)
        else:
            raise ValueError('Invalid solver type %s' % self.solver_type)

    def step(self, verbose=False):
        # clear diff
        for param in self.params:
            self._clear_diff(param)
        # load data
        for net in self.nets:
            net.load_data(None)
        # forward and get losses
        net_loss = list()
        for net in self.nets:
            loss = list()
            net.forward()
            for loss_blob in net.output_blobs:
                loss.append(loss_blob.gpu_data.get())
            net_loss.append(loss)
        # backward
        for net in self.nets:
            net.backward()
        # update
        for (param, hist1, hist2) in zip(self.params, self.historys, self.history2s):
            self._update(param, hist, hist2)
        return net_loss
