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

def create_dcgan_g(randsize=100, batchsize=64, output_chn=3, output_size=64,
                   depth=4, last_chn=128, kernel_size=4, stride=2, pad=1,
                   filler_spec=('msra', 0, 1), durelu=False):
    net = caffe_pb2.NetParameter()
    net.name = 'GenNet%d' % (randsize)
    chn = range(depth)
    chn[-1] = last_chn
    first_feat_size = output_size / 2
    for i in range(depth-1, 0, -1):
        chn[i-1] = chn[i]*2
        first_feat_size /= 2
    layers = []
    # rand vector
    layers.extend(ld.Input('rand_input', ['rand_input'],
                           [[batchsize, randsize]], 'train'))
    layers.extend(ld.Input('rand_input', ['rand_input'],
                           [[batchsize, randsize]], 'test'))
    # first layer
    num_ip = chn[0] * first_feat_size * first_feat_size
    layers.extend(ld.Linear('ip', layers[-1].top[0], num_ip, bias_term=False,
                            filler_spec=filler_spec))
    layers.extend(ld.Reshape('ip_reshape', layers[-1].top[0],
                             [batchsize, chn[0], first_feat_size, first_feat_size]))
    layers.extend(ld.Act('ip_act', layers[-1].top[0], bn_frac=0.9, durelu=durelu))
    # following layer
    for i in range(1, depth):
        layers.extend(ld.Deconv('deconv%d' % (i-depth), layers[-1].top[0],
                                chn[i], kernel_size, stride, pad,
                                bias_term=False, filler_spec=filler_spec))
        layers.extend(ld.Act('deconv%d_act' % (i-depth), layers[-1].top[0],
                             bn_frac=0.9, durelu=durelu))
    # output layer
    layers.extend(ld.Deconv('gen_data', layers[-1].top[0], output_chn,
                            kernel_size, stride, pad, bias_term=True,
                            filler_spec=filler_spec))
    layers.extend(ld.Tanh('gen_tanh', layers[-1].top[0]))
    net.force_backward = True
    net.layer.extend(layers)
    return net

def create_dcgan_d(batchsize=64, input_chn=3, input_size=64, depth=4,
                   first_chn=128, kernel_size=4, stride=2, pad=1,
                   leaky=0, sigmoid=False, filler_spec=('msra', 0, 1),
                   bnfirst=True, durelu=False):
    net = caffe_pb2.NetParameter()
    net.name = 'DisNet'
    chn = range(depth)
    chn[0] = first_chn
    for i in range(1, depth):
        chn[i] = chn[i-1] * 2
    layers = []
    # input
    layers.extend(ld.Input('input', ['input'],
                           [[batchsize, input_chn, input_size, input_size]], 'train'))
    layers.extend(ld.Input('input', ['input'],
                           [[batchsize, input_chn, input_size, input_size]], 'test'))
    # conv layers
    layers.extend(ld.Conv('conv0', layers[-1].top[0], chn[0], kernel_size,
                          stride, pad, bias_term=False, filler_spec=filler_spec))
    # let the gradient propagate to the input blob
    layers[-1].propagate_down.extend(True)
    layers.extend(ld.Act('conv0_act', layers[-1].top[0],
                         bn_frac=0.9, leaky=leaky, bn=bnfirst, durelu=durelu))
    for i in range(1, depth):
        layers.extend(ld.Conv('conv%d' % (i), layers[-1].top[0], chn[i],
                              kernel_size, stride, pad, bias_term=False,
                              filler_spec=filler_spec))
        layers.extend(ld.Act('conv%d_act' % (i), layers[-1].top[0],
                             bn_frac=0.9, leaky=leaky, durelu=durelu))
    # output
    if sigmoid:
        layers.extend(ld.Linear('ip_score', layers[-1].top[0], 1,
                                bias_term=True, filler_spec=filler_spec))
        layers.extend(ld.Sigmoid('sigmoid', layers[-1].top[0]))
    else:
        layers.extend(ld.Linear('ip_score', layers[-1].top[0], 1,
                                bias_term=False, filler_spec=filler_spec))
    layers.extend(ld.SumLoss('sum_score', layers[-1].top[0]))
    net.layer.extend(layers)
    return net

def main(args):
    if args.gstd < 0:
        filler_spec = ('msra', 0, 1)
    else:
        filler_spec = ('gaussian', 0, args.gstd)
    if args.gd == 'g':
        dcnet = create_dcgan_g(last_chn=args.chn, filler_spec=filler_spec, durelu=args.durelu)
    else:
        dcnet = create_dcgan_d(first_chn=args.chn, sigmoid=args.sigm, filler_spec=filler_spec,
                               leaky=args.leaky, bnfirst=(not args.nobnfirst), durelu=args.durelu)
    if args.output is None:
        args.output = osp.join(osp.dirname(__file__), 
                               'dcgan_g_spec.prototxt')
    with open(args.output, 'w') as f:
        f.write(pb.text_format.MessageToString(dcnet))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-o', '--output', type=str)
    parser.add_argument('--gd', type=str, choices=['g', 'd'])
    parser.add_argument('--sigm', action='store_true', default=False)
    parser.add_argument('--nobnfirst', action='store_true', default=False)
    parser.add_argument('--gstd', type=float, default=-1)
    parser.add_argument('--leaky', type=float, default=0)
    parser.add_argument('--durelu', action='store_true', default=False)
    parser.add_argument('--chn', type=int, default=128)
    args = parser.parse_args()
    main(args)
