# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 13:45:08 2017

@author: shiwu_001
"""

import os
import sys
LATTE_PATH = r'E:\projects\cpp\caffe-windows-ms\examples'
if LATTE_PATH not in sys.path:
    sys.path.insert(0, LATTE_PATH)
from latte_v2 import set_device, Blob, Net, TRAIN, TEST, \
        RandDataLoader, ImageTransformer, ImageDataLoaderPrefetch, \
        SolverBoundaryEquilibriumGAN

import os.path as osp
from argparse import ArgumentParser
import numpy as np
import time

from test_gan_util import display_samples, generate_test_samples, gen_seed, snapshot
            
def setup_solver(args):
    if args.offset:
        scale = np.array([64.0 / ((218.0 + 128.0) * 0.5)])
    else:
        scale = np.array([64.0 / 128.0])
    mean = np.array([127.5, 127.5, 127.5])
    std = np.array([127.5, 127.5, 127.5])
    mirror = True
    center = True
    seed = gen_seed()
    # generator
    netg = Net(args.netg, TRAIN)
    rdl = RandDataLoader(rand_type=args.rtype, std=args.rstd, seed=seed.next())
    netg.set_dataloader(rdl)
    # discriminator
    netd = Net(args.netd, TRAIN)
    input_dims = dict()
    for b in netd.inputs:
        input_dims[b] = netd.blobs[b].data.shape
    itf = ImageTransformer(input_dims, seed=seed.next())
    input_name = netd.inputs[0]
    itf.set_scale(input_name, scale)
    itf.set_mean(input_name, mean)
    itf.set_std(input_name, std)
    itf.set_mirror(input_name, mirror)
    itf.set_center(input_name, center)
    idl = ImageDataLoaderPrefetch(queue_size=args.qsize,
            folder=args.imfd, names=args.imnm, transformer=itf, seed=seed.next())
    for b in netd.inputs:
        idl.add_prefetch_process(b, input_dims[b])
    netd.set_dataloader(idl)
    # test nets
    test_netg = Net(args.netg, TRAIN)
    test_netg.set_dataloader(rdl)
    test_netg.share_with(netg)
    test_netd = Net(args.netd, TRAIN)
    test_netd.set_dataloader(idl)
    test_netd.share_with(netd)
    # solver
    solver = SolverBoundaryEquilibriumGAN(netg, netd, args.lrg/args.l1norm, args.lrd/args.l1norm,
                                          args.gamma, test_nets=[test_netg, test_netd],
                                          control_lr=args.control_lr/args.l1norm)
    return solver

def train(args, solver):
    netg = solver.nets[solver.netg_idx]
    netd = solver.nets[solver.netd_idx]
    test_netg = solver.test_nets[0]
    test_netd = solver.test_nets[1]
    score_blob = Blob(test_netd.blobs[args.score_blob], copy=False)
    itf = netd.dataloader.transformer
    display_iter = args.dpit
    max_iter = args.mxit
    test_iter = args.tsit
    snapshot_iter = args.snit
    image_blob_name = netd.inputs[0]
    
    loss_d_list = list()
    loss_g_list = list()
    conv_measure_list = list()
    control_k_list = list()
    start_time = time.time()
    for itr in xrange(max_iter):
        [loss_d, loss_g] = solver.step()
        conv_measure = solver.get_convergence_measure(loss_d)
        control_k = solver.control_k
        loss_d_list.append(loss_d)
        loss_g_list.append(loss_g)
        control_k_list.append(control_k)
        conv_measure_list.append(conv_measure)
        if itr % display_iter == 0:
            end_time = time.time()
            print "[%d](%.2f)" % (itr, end_time - start_time),
            start_time = end_time
            print "d: %.4e | r: %.4e | g: %.4e | k: %.4e | m: %.4e" % (
                    loss_d[2], loss_d[0], loss_d[1], control_k, conv_measure)
        if test_iter > 0 and itr % test_iter == 0:
            # generate test image
            samples = generate_test_samples(test_netg, test_netd, score_blob=score_blob)
            # deprocess image
            real_images = samples[0][0]
            real_scores = samples[0][1]
            gen_images = samples[1][0]
            gen_scores = samples[1][1]
            for i in xrange(real_images.shape[0]):
                real_images[i,...] = itf.deprocess(image_blob_name, real_images[i,...])
                gen_images[i,...] = itf.deprocess(image_blob_name, gen_images[i,...])
            if args.submean:
                mean = np.mean([real_scores, gen_scores])
                real_scores -= mean
                gen_scores -= mean
            save_path = osp.join(args.svto, '%s_%d.png' % (args.netg, itr))
            display_samples(samples, save_path)
            end_time = time.time()
            print 'Test (%.2f)' % (end_time - start_time),
            print 'save to', save_path
            start_time = end_time
        if snapshot_iter > 0 and itr % snapshot_iter == 0:
            log_obj = {'loss_d': loss_d_list,
                       'loss_g': loss_g_list,
                       'control_k': control_k_list,
                       'conv_measure': conv_measure_list}
            print 'snapshot to ', args.svto
            snapshot(itr, [netd, netg], [args.netd, args.netg], log_obj, 'log', folder=args.svto)
    # Finally snapshot
    log_obj = {'loss_d': loss_d_list,
               'loss_g': loss_g_list,
               'control_k': control_k_list,
               'conv_measure': conv_measure_list}
    print 'snapshot to ', args.svto
    snapshot(max_iter, [netd, netg], [args.netd, args.netg], log_obj, 'log', folder=args.svto)
    
def main(args):
    set_device(args.dvid)
    solver = setup_solver(args)
    if not osp.exists(args.svto):
        os.makedirs(args.svto)
    try:
        train(args, solver)
    finally:
        solver.nets[solver.netd_idx].dataloader.clean_and_close()
        print 'Queue successfully closed. Exiting program.'

def get_parser():
    ps = ArgumentParser()
    ps.add_argument('--netg', type=str)
    ps.add_argument('--netd', type=str)
    ps.add_argument('--lrg', type=float, default=0.0001)
    ps.add_argument('--lrd', type=float, default=0.0001)
    ps.add_argument('--gamma', type=float, default=1.0)
    ps.add_argument('--rtype', type=str, choices=['uniform', 'gaussian'], default='uniform')
    ps.add_argument('--rdim', type=int, default=100)
    ps.add_argument('--rstd', type=float, default=1/np.sqrt(3))
    ps.add_argument('--score_blob', type=str, default='score')
    ps.add_argument('--imfd', type=str)
    ps.add_argument('--imnm', type=str)
    ps.add_argument('--qsize', type=int, default=4)
    ps.add_argument('--mxit', type=int, default=50000)
    ps.add_argument('--dpit', type=int, default=10)
    ps.add_argument('--tsit', type=int, default=0)
    ps.add_argument('--snit', type=int, default=0)
    ps.add_argument('--svto', type=str, default=".")
    ps.add_argument('--dvid', type=int, default=0)
    ps.add_argument('--submean', action='store_true', default=False)
    ps.add_argument('--offset', action='store_true', default=False)
    ps.add_argument('--control_lr', type=float, default=0.001)
    ps.add_argument('--l1norm', type=float, default=3.0*64*64)
    return ps
    
if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)