# -*- coding: utf-8 -*-
"""
Created on Thu Mar 02 17:22:12 2017

@author: shiwu_001
"""

#import os.path as osp
import sys
import google.protobuf as pb
from argparse import ArgumentParser

#CAFFE_ROOT = r'E:\projects\cpp\caffe-windows-ms'
#PYCAFFE_PATH = osp.join(CAFFE_ROOT, r'Build\x64\Release\pycaffe')
from config import PYCAFFE_ROOT
if PYCAFFE_ROOT not in sys.path:
    sys.path.insert(0, PYCAFFE_ROOT)
from caffe.proto import caffe_pb2

import layer_def as ld

def create_encoder(code_dim=100, batchsize=64, input_chn=3, input_size=64,
                   depth=4, first_chn=128, kernel_size=4, stride=2, pad=1,
                   leaky=0, tanh=False, filler_spec=('msra', 0, 1),
                   bn=True, durelu=False):
    net = caffe_pb2.NetParameter()
    net.name = 'Encoder_dim%d_depth%d_o%d_c%d' % (code_dim, depth, input_size, first_chn)
    chn = range(depth)
    chn[0] = first_chn
    for i in xrange(1, depth):
        chn[i] = chn[i-1] * 2
    layers = []
    # input
    input_blob = 'input'
    layers.extend(ld.Input('input', [input_blob],
                           [[batchsize, input_chn, input_size, input_size]], 'train'))
    layers.extend(ld.Input('input', [input_blob],
                           [[batchsize, input_chn, input_size, input_size]], 'test'))
    ###################
    # encoder
    ###################
    # conv layers
    for i in xrange(depth):
        layers.extend(ld.Conv('conv%d' % (i), layers[-1].top[0], chn[i],
                              kernel_size, stride, pad, bias_term=False,
                              filler_spec=filler_spec))
        layers.extend(ld.BatchNorm('conv%d_act' % (i), layers[-1].top[0],
                                   moving_average_fraction=0.9))
        if durelu:
            layers.extend(ld.DuReLU('conv%d_act' % (i), layers[-1].top[0]))
        else:
            layers.extend(ld.ReLU('conv%d_act' % (i), layers[-1].top[0], leaky))
    # inner product, code
    layers.extend(ld.Linear('ip_code', layers[-1].top[0], code_dim, bias_term=False,
                            filler_spec=filler_spec))
    layers.extend(ld.BatchNorm('batchnorm_code', layers[-1].top[0],
                               moving_average_fraction=0.9, scale_after=False))
    ##################
    # decoder
    ##################
    chn[-1] = first_chn
    first_feat_size = (input_size + pad*2 - kernel_size)/stride + 1
    for i in xrange(depth-1, 0, -1):
        chn[i-1] = chn[i]*2
        first_feat_size = (first_feat_size + pad*2 - kernel_size)/stride + 1
    # inner product, decode
    decode_dim = chn[0] * first_feat_size * first_feat_size
    layers.extend(ld.Linear('ip_decode', layers[-1].top[0], decode_dim, bias_term=False,
                            filler_spec=filler_spec))
    layers.extend(ld.Reshape('reshape_code', layers[-1].top[0],
                             [batchsize, chn[0], first_feat_size, first_feat_size]))
    # deconv layers
    for i in xrange(1, depth):
        layers.extend(ld.Deconv('deconv%d' % (i-depth), layers[-1].top[0], chn[i],
                                kernel_size, stride, pad, bias_term=False,
                                filler_spec=filler_spec))
        layers.extend(ld.BatchNorm('deconv%d_act' % (i-depth), layers[-1].top[0],
                                   moving_average_fraction=0.9))
        if durelu:
            layers.extend(ld.DuReLU('deconv%d_act' % (i-depth), layers[-1].top[0]))
        else:
            layers.extend(ld.ReLU('deconv%d_act' % (i-depth), layers[-1].top[0], leaky))
    # output 
    layers.extend(ld.Deconv('devonv_decode', layers[-1].top[0], input_chn, kernel_size,
                            stride, pad, bias_term=True, filler_spec=filler_spec))
    if tanh is True:
        layers.extend(ld.Tanh('tanh_decode', layers[-1].top[0]))
    layers.extend(ld.L2Loss('l2_loss', [layers[-1].top[0], input_blob]))
    net.layer.extend(layers)
    return net

def main(args):
    if args.gstd < 0:
        filler_spec = ('msra', 0, 1)
    else:
        filler_spec = ('gaussian', 0, args.gstd)
    net = create_encoder(code_dim=args.cdim, first_chn=args.fchn,
                         filler_spec=filler_spec, tanh=args.tanh,
                         leaky=args.leaky, durelu=args.durelu)
    if args.output is None:
        args.output = osp.join(osp.dirname(__file__), 
                               'dcgan_g_spec.prototxt')
    with open(args.output, 'w') as f:
        f.write(pb.text_format.MessageToString(net))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-o', '--output', type=str)
    parser.add_argument('--tanh', action='store_true', default=False)
    parser.add_argument('--gstd', type=float, default=-1)
    parser.add_argument('--leaky', type=float, default=0)
    parser.add_argument('--durelu', action='store_true', default=False)
    parser.add_argument('--fchn', type=int, default=128)
    parser.add_argument('--cdim', type=int, default=100)
    args = parser.parse_args()
    main(args)
