# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 14:50:59 2017

@author: shiwu_001
"""

from solver_all import Solver
#from blob import Blob
#from math_func import setx
from config import DTYPE

class SolverBoundaryEquilibriumGAN(Solver):
    def __init__(self, netg, netd, lrg, lrd, gamma,
                 control_k=0., control_lr=0.001, alter_train=False,
                 test_nets=None, lr=0.001, decay=0.,
                 mom=0.9, mom2=0.999, eps=1e-8, solver_type='Adam'):
        super(SolverBoundaryEquilibriumGAN, self).__init__([netg, netd], test_nets,
                lr, decay, mom, mom2, eps, solver_type)
        self.lrs = [lrg, lrd]
        self.accum_mults = [0.] * 2
        self.accum_mult2s = [0.] * 2
        self.gamma = gamma
        self.control_k = control_k
        self.control_lr = control_lr
        self.netg_idx = 0
        self.netd_idx = 1
        self.netg = netg
        self.netd = netd
        self.alter_train = alter_train
        self.netg.output_blobs[0].share_data(self.netd.input_blobs[0])
        self.netg.output_blobs[0].share_diff(self.netd.input_blobs[0])
        
    def _update_by_idx(self, idx):
        self.lr = self.lrs[idx]
        self.accum_mult = self.accum_mults[idx]
        self.accum_mult2 = self.accum_mult2s[idx]
        self._update(self.params[idx], self.historys[idx], self.history2s[idx])
        self.accum_mults[idx] = self.accum_mult
        self.accum_mult2s[idx] = self.accum_mult2

    def _step_d(self, verbose=False):
        loss_d = []
        netd = self.netd
        netg = self.netg
        d_output_blob = netd.output_blobs[0]
#        d_input_blob = netd.input_blobs[0]
#        g_output_blob = netg.output_blobs[0]
        # clear diff
        self._clear_diff(self.params[self.netd_idx])
        # real data
        netd.load_data(None)
        netd.forward()
        loss_d.append(d_output_blob.gpu_data.get())
        d_output_blob.gpu_diff.fill(DTYPE(1.))
        netd.backward()
        # generated data
        netg.load_data(None)
        netg.forward()
#        setx(g_output_blob.gpu_data, d_input_blob.gpu_data)
        netd.forward()
        loss_d.append(d_output_blob.gpu_data.get())
        d_output_blob.gpu_diff.fill(DTYPE(-self.control_k))
        netd.backward()
        # update if alternates training
        if self.alter_train:
            self._update_by_idx(self.netd_idx)
        return loss_d
        
    def _step_g(self, verbose=False):
        netg = self.netg
        netd = self.netd
#        g_output_blob = netg.output_blobs[0]
#        d_input_blob = netd.input_blob[0]
        d_output_blob = netd.output_blobs[0]
        # clear diff
        self._clear_diff(self.params[self.netg_idx])
        # generated data
        netg.load_data(None)
        netg.forward()
#        setx(g_output_blob.gpu_data, d_input_blob.gpu_data)
        netd.forward()
        loss_g = d_output_blob.gpu_data.get()
        d_output_blob.gpu_diff.fill(DTYPE(1.))
        netd.backward()
#        setx(d_input_blob.gpu_diff, g_output_blob.gpu_diff)
        netg.backward()
        # update if alternates training
        if self.alter_train:
            self._update_by_idx(self.netg_idx)
        return loss_g
    
    def get_convergence_measure(self, loss_d):
        return loss_d[0] + abs(loss_d[0] * self.gamma - loss_d[1])
        
    def step(self, verbose=False):
        # step for generator
        loss_g = self._step_g(verbose)
        # step for discriminator
        loss_d = self._step_d(verbose)
        loss_d.append(loss_d[0] - self.control_k*loss_d[1])
        # update the control parameter
        self.control_k += self.control_lr * (self.gamma * loss_d[0] - loss_d[1])
        if self.control_k < 0:
            self.control_k = 0
        if self.control_k > 1:
            self.control_k = 1
        # update if not alternates training
        if not self.alter_train:
            self._update_by_idx(self.netd_idx)
            self._update_by_idx(self.netg_idx)
        return [loss_d, loss_g]

#    def solve(self, args):
#        netg = self.netg
#        netd = self.netd
#        test_netg = self.test_nets[self.netg_idx]
#        test_netd = self.test_nets[self.netd_idx]
#        max_iter = args.max_iter
#        display_iter = args.display_iter
#        test_iter = args.test_iter
#        snapshot_iter = args.snapshot_iter
#        loss_d_list = list()
#        loss_g_list = list()
#        control_k_list = list()
#        conv_measure_list = list()
#        import time
#        start_time = time.time()
#        for itr in xrange(max_iter):
#            loss_d, loss_g = self.step()
#            conv_measure = self.get_convergence_measure(loss_d)
#            loss_d_list.append(loss_d)
#            loss_g_list.append(loss_g)
#            control_k_list.append(self.control_k)
#            conv_measure_list.append(conv_measure)
#            if itr % display_iter == 0:
#                end_time = time.time()
#                print "[%d](%.2f)" % (itr, end_time - start_time),
#                start_time = end_time
#                print "d: %.4e | x: %.4e | g: %.4e | k: %.4e | m: %.4e" % (
#                        loss_d[2], loss_d[0], loss_d[1], self.control_k, conv_measure)
#            if test_iter > 0 and itr % test_iter == 0:
#                # generate test image
#                test_netg.share_with(netg)
#                test_netd.share_with(netd)
#                samples = generate_test_samples(test_netg, test_netd)
