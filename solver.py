# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 14:31:55 2016

@author: shiwu_001
"""

import numpy as np
from caffe import check_mode_gpu
from config import DTYPE
from blob import Blob
from math_func import axpby, axpbypcz, sumsqr#, setx
#import pycuda.gpuarray as garr
#import time

class MySolver(object):
    def __init__(self, net, test_net=None):
        self.net = net
        if test_net != None:
            self.test_net = test_net

class SGDSolver(MySolver):
    def __init__(self, *args, **kwargs):
        super(SGDSolver, self).__init__(*args, **kwargs)
        self.history = self._init_history()
        self.net_blobs = self._init_net_blobs()
        self.lr = 0.1
        self.mom = 0.9
        self.decay = 0.0001
        self.use_gpu = check_mode_gpu()
    
    def set_use_gpu(self, use_gpu):
        self.use_gpu = use_gpu
    
    def _init_history(self):
        """
            Init the history (momentum) of blobs
        """
        history = list()
        for l in self.net.layers:
            if l.type == "Convolution":
#                history.append([np.zeros_like(l.blobs[0].data)])
                history.append([Blob(l.blobs[0], copy=True)])
            elif l.type == "Scale" or l.type == "InnerProduct":
#                history.append([np.zeros_like(l.blobs[0].data),
#                                np.zeros_like(l.blobs[1].data)])
                history.append([Blob(l.blobs[0], copy=True),
                                Blob(l.blobs[1], copy=True)])
            else:
                history.append(None)
        return history
    
    def _init_net_blobs(self):
        """
            Expose the blobs (data/diff) of net to python. No copy.
        """
        net_blobs = list()
        for l in self.net.layers:
            if l.type == "Convolution":
                net_blobs.append([Blob(l.blobs[0])])
            elif l.type == "Scale" or l.type == "InnerProduct":
                net_blobs.append([Blob(l.blobs[0]),
                                  Blob(l.blobs[1])])
            else:
                net_blobs.append(None)
        return net_blobs
        
    def _clear_diff(self, net):
        if self.use_gpu:
            for l in self.net_blobs:
                if l is not None:
                    for j in xrange(len(l)):
#                        seta(DTYPE(0), l[j].gpu_diff)
                        l[j].gpu_diff.fill(DTYPE(0))
        else:
            for l in net.layers:
                if l.type == "Convolution":
                    l.blobs[0].diff[...] = DTYPE(0)
                elif l.type == "Scale" or l.type == "InnerProduct":
                    l.blobs[0].diff[...] = DTYPE(0)
                    l.blobs[1].diff[...] = DTYPE(0)
    
    
    def _update_ReL2_cpu(self, net, lr, mom, decay):
        history = self.history
        for i, l in enumerate(net.layers):
            if l.type == "Convolution":
                local_lr = lr*np.sum(np.square(l.blobs[0].data))
                history[i][0].data[...] = history[i][0].data*mom + l.blobs[0].diff \
                    + l.blobs[0].data*decay
                l.blobs[0].data[...] -= local_lr*history[i][0].data
            elif l.type == "Scale" or l.type == "InnerProduct":
                local_lr = lr*np.sum(np.square(l.blobs[0].data))
                history[i][0].data[...] = history[i][0].data*mom + l.blobs[0].diff \
                    + l.blobs[0].data*decay
                history[i][1].data[...] = history[i][1].data*mom + l.blobs[1].diff \
                    + l.blobs[1].data*decay
                l.blobs[0].data[...] -= local_lr*history[i][0].data
                l.blobs[1].data[...] -= local_lr*history[i][1].data

    def _update_L2_cpu(self, net, lr, mom, decay):
        history = self.history
        for i, l in enumerate(net.layers):
            if l.type == "Convolution":
                local_lr = lr
                history[i][0].data[...] = history[i][0].data*mom + l.blobs[0].diff \
                    + l.blobs[0].data*decay
                l.blobs[0].data[...] -= local_lr*history[i][0].data
            elif l.type == "Scale" or l.type == "InnerProduct":
                local_lr = lr
                history[i][0].data[...] = history[i][0].data*mom + l.blobs[0].diff \
                    + l.blobs[0].data*decay
                history[i][1].data[...] = history[i][1].data*mom + l.blobs[1].diff \
                    + l.blobs[1].data*decay
                l.blobs[0].data[...] -= local_lr*history[i][0].data
                l.blobs[1].data[...] -= local_lr*history[i][1].data
    
    def _update_ReL2_gpu(self, net, lr, mom, decay):
        history = self.history
        net_blobs = self.net_blobs
        for i, l in enumerate(net.layers):
            if l.type in ["Convolution", "Scale", "InnerProduct"]:
                local_lr = lr * (sumsqr(net_blobs[i][0].gpu_data).get())
                for j in xrange(len(history[i])):
                    axpbypcz(mom, history[i][j].gpu_data,
                             DTYPE(1), net_blobs[i][j].gpu_diff,
                             decay, net_blobs[i][j].gpu_data,
                             history[i][j].gpu_data)
                    axpby(DTYPE(1), net_blobs[i][j].gpu_data,
                          -local_lr, history[i][j].gpu_data,
                          net_blobs[i][j].gpu_data)
        
    def _update_L2_gpu(self, net, lr, mom, decay):
        history = self.history
        net_blobs = self.net_blobs
        for i, l in enumerate(net.layers):
            if l.type in ["Convolution", "Scale", "InnerProduct"]:
                local_lr = lr
                for j in xrange(len(history[i])):
                    axpbypcz(mom, history[i][j].gpu_data,
                             DTYPE(1), net_blobs[i][j].gpu_diff,
                             decay, net_blobs[i][j].gpu_data,
                             history[i][j].gpu_data)
                    axpby(DTYPE(1), net_blobs[i][j].gpu_data,
                          -local_lr, history[i][j].gpu_data,
                          net_blobs[i][j].gpu_data)
    
    def _update_ReL2(self, net, lr, mom, decay):
        if self.use_gpu:
            self._update_ReL2_gpu(net, lr, mom, decay)
        else:
            self._update_ReL2_cpu(net, lr, mom, decay)
    
    def _update_L2(self, net, lr, mom, decay):
        if self.use_gpu:
            self._update_L2_gpu(net, lr, mom, decay)
        else:
            self._update_L2_cpu(net, lr, mom, decay)
    
    def update(self, net, lr, mom, decay, dist_type):
        if dist_type == 'ReL2':
            self._update_ReL2(net, lr, mom, decay)
        elif dist_type == 'L2':
            self._update_L2(net, lr, mom, decay)
        
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
        output_fw_ = self.net.forward()
#        etime = time.time()
#        print "forw %.3f|" % ((etime - stime)*100),
#        stime = etime
        output_bw_ = self.net.backward()
#        etime = time.time()
#        print "bckw %.3f|" % ((etime - stime)*100),
#        stime = etime
        self.update(self.net, lr, mom, decay, dist_type)
#        etime = time.time()
#        print "updt %.3f|" % ((etime - stime)*100)
        return output_fw_, output_bw_, batchid_
        
    def step_n(self, n, lr, mom, decay, dist_type, verbose=False):
        avg = None
        for i in xrange(n):
            ofw, _, _ = self.step(lr, mom, decay, dist_type, verbose)
            if avg == None:
                avg = dict()
                for k,v in ofw.items():
                    avg[k] = np.copy(v)
            else:
                for k,v in ofw.items():
                    avg[k] += v
        for k in avg.keys():
            avg[k] /= n
        return avg
    
    def test(self):
        net = self.test_net
        batchsize_ = net.blobs[net.dataloader.data_blob].num
        datasize_ = net.dataloader.nimages
        assert(datasize_ % batchsize_ == 0)
        loss = 0.
        top1 = 0.
        for start_id in xrange(0, datasize_, batchsize_):
            stop_id = start_id + batchsize_ if start_id + batchsize_ < datasize_ else datasize_
            batchid_ = np.arange(start_id, stop_id)
            net.load_data(batchid_)
            ofw = net.forward()
            loss += ofw['loss']
            top1 += ofw['accuracy_top1']
        loss /= (datasize_ / batchsize_)
        top1 /= (datasize_ / batchsize_)
        return loss, top1
    
#    def init_blobs_act(self, net, copy=True):
#        act = list()
#        for i, l in enumerate(net.layers):
#            if l.type == "ReLU":
#                blob_ids = net._top_ids(i)
#                blob_name = net._blob_names[blob_ids[0]]
#                act.append(Blob(net.blobs[blob_name], copy=copy))
#        return act
#    
#    def copy_blobs_act_gpu(self, blobs0, blobs1):
#        '''
#            blobs1[i].gpu_data = blobs0[i].gpu_data
#        '''
#        for b0,b1 in zip(blobs0, blobs1):
#            setx(b0.gpu_data, b1.gpu_data)
#        
#    def get_trans_gpu(self, blob0, blob1):
#    #    return garr.dot(blob0.gpu_data, blob1.gpu_data).get()
##        return sumxorpos(blob0.gpu_data, blob1.gpu_data)
#        xorpos(blob0.gpu_data, blob1.gpu_data, blob0.gpu_data)
#        return garr.sum(blob0.gpu_data).get()
    
    def solve(self):
        pass