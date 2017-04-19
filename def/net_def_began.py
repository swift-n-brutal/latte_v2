# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 15:16:13 2017

@author: shiwu_001
"""

import os.path as osp
import sys
import google.protobuf as pb
from argparse import ArgumentParser
import numpy as np

from config import PYCAFFE_ROOT
if PYCAFFE_ROOT  not in sys.path:
    sys.path.insert(0, PYCAFFE_ROOT)
from caffe.proto import caffe_pb2

import layer_def as ld

def create_encoder_decoder(code_dim=128, batchsize=16, input_chn=3, input_size=64,
                   depth=4, repeat=2, first_chn=128, kernel_size=3, stride=2, pad=1,
                   leaky=0, alpha=1.0, tanh=False, filler_spec=('msra', 0, 1), 
                   bn=False, durelu=False):
    net = caffe_pb2.NetParameter()
    net.name = 'Encoder_decoder_dim%d_depth%d_repeat%d_o%d_c%d' % \
            (code_dim, depth, repeat, input_size, first_chn)
    layers = []
    chn = range(depth)
    for i in xrange(depth):
        chn[i] = first_chn * i + first_chn
    # input
    input_blob = 'input'
    layers.extend(ld.Input('input', [input_blob],
                           [[batchsize, input_chn, input_size, input_size]], 'train'))
    layers.extend(ld.Input('input', [input_blob],
                           [[batchsize, input_chn, input_size, input_size]], 'test'))
    ################
    # encoder
    ################
    # conv layers  stride
    for i in xrange(depth):
        for j in xrange(repeat - 1):
            lname = "conv%d_%d" % (i, j)
            layers.extend(ld.Conv(lname, layers[-1].top[0], chn[i],
                                  kernel_size, 1, 1, bias_term=True, filler_spec=filler_spec))
            layers.extend(ld.ELU(lname+'_act', layers[-1].top[0], alpha=alpha))
        lname = "conv%d_%d" % (i, repeat-1)
        if i < depth - 1:
            layers.extend(ld.Conv(lname, layers[-1].top[0], chn[i],
                                  kernel_size, stride, 1, bias_term=True, filler_spec=filler_spec))
        else:
            layers.extend(ld.Conv(lname, layers[-1].top[0], chn[i],
                                  kernel_size, 1, 1, bias_term=True, filler_spec=filler_spec))
        layers.extend(ld.ELU(lname+'_act', layers[-1].top[0], alpha=alpha))
    # inner product, code
    layers.extend(ld.Linear('ip_code', layers[-1].top[0], code_dim, bias_term=True,
                            filler_spec=filler_spec))
    #################
    # decoder
    #################
    first_feat_size = input_size
    for i in xrange(depth):
        chn[i] = first_chn
        if i > 0:
            first_feat_size = (first_feat_size + pad*2 - kernel_size)/stride + 1
    assert first_feat_size == 8, 'The first feature size should be 8 (%d given)' % first_feat_size
    # inner product, decode
    decode_dim = [batchsize, chn[0], first_feat_size, first_feat_size]
    layers.extend(ld.Linear('ip', layers[-1].top[0], np.prod(decode_dim[1:]),
                            bias_term=True, filler_spec=filler_spec))
    layers.extend(ld.Reshape('reshape_ip', layers[-1].top[0], decode_dim))
    # conv layers + upsample
    for i in xrange(depth):
        for j in xrange(repeat):
            lname = "conv%d_%d" % (i - depth, j)
            layers.extend(ld.Conv(lname, layers[-1].top[0], chn[i],
                                  kernel_size, 1, 1, bias_term=True, filler_spec=filler_spec))
            layers.extend(ld.ELU(lname+'_act', layers[-1].top[0], alpha=alpha))
        if i < depth - 1:
            layers.extend(ld.Upsample("upsample%d" % (i-depth), layers[-1].top[0], factor=stride))
    # output image
    output_blob = 'output'
    layers.extend(ld.Conv(output_blob, layers[-1].top[0], input_chn, kernel_size,
                          1, 1, bias_term=True, filler_spec=filler_spec))
    if tanh:
        layers.extend(ld.Tanh('tanh_'+output_blob, layers[-1].top[0]))
    # loss
    layers.extend(ld.L1Loss('l1_loss', [output_blob, input_blob], samplewise=True))
    net.layer.extend(layers)
    return net
    
def create_generator(code_dim=128, batchsize=16, input_chn=3, input_size=64,
                   depth=4, repeat=2, first_chn=128, kernel_size=3, stride=2, pad=1,
                   leaky=0, alpha=1.0, tanh=False, filler_spec=('msra', 0, 1), 
                   bn=False, durelu=False):
    net = caffe_pb2.NetParameter()
    net.name = 'Generator_dim%d_depth%d_repeat%d_o%d_c%d' % \
            (code_dim, depth, repeat, input_size, first_chn)
    layers = []
    chn = range(depth)
    first_feat_size = input_size
    for i in xrange(depth):
        chn[i] = first_chn
        if i > 0:
            first_feat_size = (first_feat_size + pad*2 - kernel_size)/stride + 1
    assert first_feat_size == 8, 'The first feature size should be 8 (%d given)' % first_feat_size
    # input
    input_blob = 'input'
    layers.extend(ld.Input('input', [input_blob],
                           [[batchsize, code_dim]], 'train'))
    layers.extend(ld.Input('input', [input_blob],
                           [[batchsize, code_dim]], 'test'))
    # inner product, decode
    decode_dim = [batchsize, chn[0], first_feat_size, first_feat_size]
    layers.extend(ld.Linear('ip', layers[-1].top[0], np.prod(decode_dim[1:]),
                            bias_term=True, filler_spec=filler_spec))
    layers.extend(ld.Reshape('reshape_ip', layers[-1].top[0], decode_dim))
    # conv layers + upsample
    for i in xrange(depth):
        for j in xrange(repeat):
            lname = "conv%d_%d" % (i - depth, j)
            layers.extend(ld.Conv(lname, layers[-1].top[0], chn[i],
                                  kernel_size, 1, 1, bias_term=True, filler_spec=filler_spec))
            layers.extend(ld.ELU(lname+'_act', layers[-1].top[0], alpha=alpha))
        if i < depth - 1:
            layers.extend(ld.Upsample("upsample%d" % (i-depth), layers[-1].top[0], factor=stride))
    # output image
    output_blob = 'output'
    layers.extend(ld.Conv(output_blob, layers[-1].top[0], input_chn, kernel_size,
                          1, 1, bias_term=True, filler_spec=filler_spec))
    if tanh:
        layers.extend(ld.Tanh('tanh_'+output_blob, layers[-1].top[0]))
    net.force_backward = True
    net.layer.extend(layers)
    return net

def main(args):
    if args.gstd < 0:
        filler_spec = ('msra', 0, 1)
    else:
        filler_spec = ('gaussian', 0, args.gstd)
    if args.gd == 'g':
        net = create_generator(code_dim=args.cdim, batchsize=args.bsize,
                               input_size=args.isize, first_chn=args.fchn,
                               depth=args.depth, repeat=args.repeat,
                               leaky=args.leaky, tanh=args.tanh,
                               filler_spec=filler_spec)
    else:
        net = create_encoder_decoder(code_dim=args.cdim, batchsize=args.bsize,
                               input_size=args.isize, first_chn=args.fchn,
                               depth=args.depth, repeat=args.repeat,
                               leaky=args.leaky, tanh=args.tanh,
                               filler_spec=filler_spec)
    if args.output is None:
        args.output = osp.join(osp.dirname(__file__),
                               'began_%c_spec.prototxt' % args.gd)
    with open(args.output, 'w') as f:
        f.write(pb.text_format.MessageToString(net))

def get_parser():
    parser = ArgumentParser()
    parser.add_argument('-o', '--output', type=str)
    parser.add_argument('--gd', type=str, choices=['g', 'd'])
    parser.add_argument('--cdim', type=int, default=128)
    parser.add_argument('--bsize', type=int, default=16)
    parser.add_argument('--isize', type=int, default=64)
    parser.add_argument('--fchn', type=int, default=128)
    parser.add_argument('--repeat', type=int, default=2)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--gstd', type=float, default=-1)
    parser.add_argument('--leaky', type=float, default=0)
    parser.add_argument('--durelu', action='store_true', default=False)
    parser.add_argument('--tanh', action='store_true', default=False)
    return parser
    
if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)