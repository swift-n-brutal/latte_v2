# -*- coding: utf-8 -*-
"""
Created on Fri Mar 03 10:40:06 2017

@author: shiwu_001
"""

import os
import os.path as osp
import sys
#import google.protobuf as pb
from argparse import ArgumentParser
import numpy as np
import time

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

LATTE_PATH = '/home/sw015/project'
if LATTE_PATH not in sys.path:
    sys.path.insert(0, LATTE_PATH)
from latte_v2 import set_device, Net, SolverWGAN, RandDataLoader, \
        ImageDataLoader, ImageTransformer, ImageDataLoaderPrefetch

#CAFFE_ROOT = r'E:\projects\cpp\caffe-windows-ms'
#PYCAFFE_PATH = osp.join(CAFFE_ROOT, r'Build\x64\Release\pycaffe')
#if PYCAFFE_PATH not in sys.path:
#    sys.path.insert(0, PYCAFFE_PATH)
from caffe import TRAIN, TEST

#print sys.path

def get_plotable_data(data):
    data[data < 0] = 0
    data[data > 255] = 255
    data = data.swapaxes(0,1).swapaxes(1,2)
    data = np.require(data, dtype=np.uint8)
    return data

def display_samples(samples, transformer, blob, save_path,
                    stride=2, font_size=10, text_height=12,
                    font_name='/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'):
    real_images = samples[0][0]
    real_scores = samples[0][1]
    gen_images = samples[1][0]
    gen_scores = samples[1][1]
    n_samples = real_images.shape[0]
    rows = int(np.sqrt(n_samples))
    cols = (n_samples + rows - 1)  / rows
    im_h = real_images.shape[2]
    im_w = real_images.shape[3]
    font = ImageFont.truetype(font_name, size=font_size)
    canvas = Image.new('RGB', ((im_w + stride)*cols*2,
                               (im_h + text_height)*rows),
                               'white')
    draw = ImageDraw.Draw(canvas)
    for num in xrange(n_samples):
        # index to put the images
        i = num / cols
        j = num - i*cols
        # real image
        im = transformer.deprocess(blob, real_images[num,...])
        im = Image.fromarray(get_plotable_data(im))
        score = real_scores[num,0]
        sgn = 'green' if score > 0 else 'red'
        x = (im_w + stride) * j
        y = (im_h + text_height) * i
        canvas.paste(im, (x,y))
        draw.text((x, y + im_h), '%.4e' % score, font=font, fill=sgn)
        # generated sample
        im = transformer.deprocess(blob, gen_images[num,...])
        im = Image.fromarray(get_plotable_data(im))
        score = gen_scores[num,0]
        sgn = 'green' if score > 0 else 'red'
        x = x + (im_w + stride)*cols
#        y = (im_h + text_height) * i
        canvas.paste(im, (x,y))
        draw.text((x, y + im_h), '%.4e' % score, font=font, fill=sgn)
    canvas.save(save_path, 'PNG')

def generate_test_samples(net_g, net_d, n_samples=64):
    gen_blob = net_g.outputs[0]
    image_blob = net_d.inputs[0]
    score_blob = 'ip_score'
    real_images = []
    real_scores = []
    gen_images = []
    gen_scores = []
    count = 0
    batchsize = net_d.blobs[image_blob].shape[0]
    while count < n_samples:
        # generate images
        net_g.load_data(None)
        ofw_g = net_g.forward(no_output=False)
        gen_images.append(np.copy(ofw_g[gen_blob]))
        # get scores of generated images
        net_d.blobs[image_blob].data[...] = ofw_g[gen_blob][...]
        net_d.forward()
        gen_scores.append(np.copy(net_d.blobs[score_blob].data))
        # load real images
        net_d.load_data(None)
        real_images.append(np.copy(net_d.blobs[image_blob].data))
        net_d.forward()
        real_scores.append(np.copy(net_d.blobs[score_blob].data))
        count += batchsize
    real_images = np.concatenate(real_images)
    real_scores = np.concatenate(real_scores)
    gen_images = np.concatenate(gen_images)
    gen_scores = np.concatenate(gen_scores)
    return [(real_images[:n_samples,...], real_scores[:n_samples,...]),
            (gen_images[:n_samples,...], gen_scores[:n_samples,...])]

def setup_solver(args):
    scale = np.array([64.0 / ((218.0 + 128.0) *0.5)])
    mean = np.array([127.5, 127.5, 127.5])
    std = np.array([127.5, 127.5, 127.5])
    rand_std = 1.0
    mirror = True
    center = False
    # generator
    net_g = Net(args.netg, TRAIN)
    net_g.set_data_blobs(net_g.inputs)
    rdl = RandDataLoader(std=rand_std, seed=37)
    net_g.set_dataloader(rdl)
    # discriminator
    net_d = Net(args.netd, TRAIN)
    net_d.set_data_blobs(net_d.inputs)
        # input transformer
    input_dims = dict()
    for b in net_d.inputs:
        input_dims[b] = net_d.blobs[b].data.shape
    itf = ImageTransformer(input_dims, seed=137)
    itf.set_scale(net_d.inputs[0], scale)
    itf.set_mean(net_d.inputs[0], mean)
    itf.set_std(net_d.inputs[0], std)
    itf.set_mirror(net_d.inputs[0], mirror)
    itf.set_center(net_d.inputs[0], center)
        # image data loader
    idl = ImageDataLoaderPrefetch(queue_size=args.qusz,
            folder=args.imfd, names=args.imnm, transformer=itf, seed=237)
    net_d.set_dataloader(idl)
    for b in net_d.inputs:
        idl.add_prefetch_process(b, input_dims[b])
    # test nets
    test_net_g = Net(args.netg, TRAIN)
    test_net_g.set_data_blobs(test_net_g.inputs)
    test_net_g.set_dataloader(rdl)
    test_net_d = Net(args.netd, TRAIN)
    test_net_d.set_data_blobs(test_net_d.inputs)
    test_net_d.set_dataloader(idl)
    # solver
    solver = SolverWGAN(net_g, net_d, test_net_g, test_net_d)
    return solver

def train(args, solver):
    net_g = solver.net_g
    net_d = solver.net_d
    test_net_g = solver.test_net_g
    test_net_d = solver.test_net_d
    itf = net_d.dataloader.transformer
    display_iter = args.dpit
    max_iter = args.mxit
    test_iter = args.tsit
    snapshot_iter = args.snap
    start_time = time.time()
    loss_d_list = []
    loss_g_list = []
    for itr in xrange(max_iter):
        if args.fancy and (itr < 25 or itr % args.fancy_step == 0):
            solver.inner_iter = args.fancy_inner
        else:
            solver.inner_iter = args.inner
        solver.lr = args.lr
        [loss_d, loss_g] = solver.step()
        loss_d_list.append(loss_d)
        loss_g_list.append(loss_g)
        if itr % display_iter == 0:
            end_time = time.time()
            print '[%05d](%.2f)' % (itr, end_time - start_time),
            start_time = end_time
            print 'g: %.4f |' % loss_g,
            print 'd: %s' % (' '.join(['(%.4f,%.4f)' % (l[0], l[1]) for l in loss_d]))
        if test_iter > 0 and itr % test_iter == 0:
            # generate test image
            test_net_g.share_with(net_g)
            test_net_d.share_with(net_d)
            samples = generate_test_samples(test_net_g, test_net_d)
            if not osp.exists(args.svto):
                os.makedirs(args.svto)
            save_path = osp.join(args.svto, '%s_%05d.png' % (args.netg, itr))
            display_samples(samples, itf, net_d.inputs[0], save_path)
            end_time = time.time()
            print 'Test(%.2f)' % (end_time - start_time),
            print 'save to', save_path
        if snapshot_iter > 0 and itr % snapshot_iter == 0:
            log_obj = {'loss_d': loss_d_list,
                       'loss_g': loss_g_list}
            print 'snapshot to ', args.svto
            save_snapshot(itr, osp.join(args.svto, 'snapshot'), solver, log_obj)
    # Finally snapshot
    log_obj = {'loss_d': loss_d_list,
               'loss_g': loss_g_list}
    print 'snapshot to ', args.svto
    save_snapshot(max_iter, osp.join(args.svto, 'snapshot'), solver, log_obj)
                
def save_snapshot(itr, prefix, solver, log_obj):
    solver.net_d.save('%s_d_itr_%d.caffemodel' % (prefix, itr))
    solver.net_g.save('%s_g_itr_%d.caffemodel' % (prefix, itr))
    name = prefix + '.npz'
    eval("np.savez('%s', %s)" % (name, ','.join(["%s=log_obj['%s']" % (k,k) for k in log_obj.keys()])))
    
def main(args):
    set_device(args.dvid)
    solver = setup_solver(args)
    try:
        train(args, solver)
    finally:
        solver.net_d.dataloader.clean_and_close()
        print 'Queue successfully closed. Exiting program.'
    
def get_parser():
    ps = ArgumentParser()
#    parser.add_argument('-o', '--output', type=str)
    ps.add_argument('--lr', type=float, default=0.00005)
    ps.add_argument('--netg', type=str)
    ps.add_argument('--netd', type=str)
    ps.add_argument('--rdim', type=int, default=100)
    ps.add_argument('--imfd', type=str, default='')
    ps.add_argument('--imnm', type=str, default='')
    ps.add_argument('--mxit', type=int, default=10000)
    ps.add_argument('--dpit', type=int, default=10)
    ps.add_argument('--tsit', type=int, default=0)
    ps.add_argument('--svto', type=str, default=".")
    ps.add_argument('--qusz', type=int, default=2)
    ps.add_argument('--snap', type=int, default=0)
    ps.add_argument('--dvid', type=int, default=0)
    ps.add_argument('--inner', type=int, default=5)
    ps.add_argument('--fancy', action='store_true', default=False)
    ps.add_argument('--fancy_inner', type=int, default=100)
    ps.add_argument('--fancy_step', type=int, default=500)
    return ps
    
if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
    
