# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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

#CAFFE_ROOT = r'E:\projects\cpp\caffe-windows-ms'
#PYCAFFE_PATH = osp.join(CAFFE_ROOT, r'Build\x64\Release\pycaffe')
#if PYCAFFE_PATH not in sys.path:
#    sys.path.insert(0, PYCAFFE_PATH)
from caffe import TRAIN, TEST

LATTE_PATH = '/home/sw015/project'
if LATTE_PATH not in sys.path:
    sys.path.insert(0, LATTE_PATH)
from latte_v2 import set_device, Net, SolverWGAN, RandDataLoader, \
        ImageDataLoader, ImageTransformer, ImageDataLoaderPrefetch, \
        Blob, math_func
        
def main():
    set_device(1)
    qusz = 4
    imfd = r'E:\images\img_align_celeba'
    imnm = r'E:\images\img_align_celeba\names.txt'
    mean = np.array([127.5, 127.5, 127.5])
    std = np.array([127.5, 127.5, 127.5])
    mirror = True
    center = True
    rand_std = 1.0
    # generator
    net_g = Net('dcgan_g_d4_c128_o64_std0.02.prototxt', TRAIN)
    net_g.set_data_blobs(net_g.inputs)
    rdl = RandDataLoader(std=rand_std, seed=37)
    net_g.set_dataloader(rdl)
    # discriminator
    net_d = Net('dcgan_d_d4_c128_o64_std0.02_leaky0.2.prototxt', TRAIN)
    net_d.set_data_blobs(net_d.inputs)
        # input transformer
    input_dims = dict()
    for b in net_d.inputs:
        input_dims[b] = net_d.blobs[b].data.shape
    itf = ImageTransformer(input_dims, seed=137)
    itf.set_mean(net_d.inputs[0], mean)
    itf.set_std(net_d.inputs[0], std)
    itf.set_mirror(net_d.inputs[0], mirror)
    itf.set_center(net_d.inputs[0], center)
    try:
            # image data loader
        idl = ImageDataLoaderPrefetch(queue_size=qusz,
                folder=imfd, names=imnm, transformer=itf, seed=237)
        for b in net_d.inputs:
            idl.add_prefetch_process(b, input_dims[b])
        net_d.set_dataloader(idl)
        net_g.load_data(None)
        net_g.forward()
        gen_blob = Blob(net_g.blobs[net_g.outputs[0]], copy=False)
        image_blob = Blob(net_d.blobs[net_d.inputs[0]], copy=False)
        score_blob = Blob(net_d.blobs[net_d.outputs[0]], copy=False)
        ip_blob = Blob(net_g.blobs['ip'], copy=False)
        math_func.setx(gen_blob.gpu_data, image_blob.gpu_data)
        net_d.forward()
        score_blob.gpu_diff.fill(np.float32(-1.0))
        net_d.backward()
        math_func.setx(image_blob.gpu_diff, gen_blob.gpu_diff)
        net_g.backward()
        print 'gen_blob',
        print math_func.sumsqr(gen_blob.gpu_data).get(), math_func.sumsqr(gen_blob.gpu_diff).get()
        print gen_blob.gpu_data.get()[0,0,...], gen_blob.gpu_diff.get()[0,0,...]
        print 'image_blob',
        print math_func.sumsqr(image_blob.gpu_data).get(), math_func.sumsqr(image_blob.gpu_diff).get()
        print image_blob.gpu_data.get()[0,0,...], image_blob.gpu_diff.get()[0,0,...]
        print 'score_blob',
        print score_blob.data, score_blob.diff
        print 'ip_blob',
        print math_func.sumsqr(ip_blob.gpu_data).get(), math_func.sumsqr(ip_blob.gpu_diff).get()
    finally:
        net_d.dataloader.clean_and_close()
        print 'Queue successfully closed. Exiting program.'
    return 0
    
if __name__ == '__main__':
    main()
