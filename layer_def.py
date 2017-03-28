# -*- coding: utf-8 -*-
"""
Created on Thu Mar 02 15:01:51 2017

@author: shiwu_001
"""

#import os.path as osp
#import sys
#import google.protobuf as pb

#CAFFE_ROOT = r'E:\projects\cpp\caffe-windows-ms'
#PYCAFFE_PATH = osp.join(CAFFE_ROOT, r'Build\x64\Release\pycaffe')
#if PYCAFFE_PATH not in sys.path:
#    sys.path.insert(0, PYCAFFE_PATH)
from caffe.proto import caffe_pb2

def _get_include(phase):
    inc = caffe_pb2.NetStateRule()
    if phase == 'train':
        inc.phase = caffe_pb2.TRAIN
    elif phase == 'test':
        inc.phase = caffe_pb2.TEST
    else:
        raise ValueError("Unknown phase {}".format(phase))
    return inc
    
def _get_param(num_param):
    if num_param == 1:
        param = caffe_pb2.ParamSpec()
        param.lr_mult = 1
        param.decay_mult = 1
        return [param]
    elif num_param == 2:
        param_w = caffe_pb2.ParamSpec()
        param_w.lr_mult = 1
        param_w.decay_mult = 1
        param_b = caffe_pb2.ParamSpec()
        param_b.lr_mult = 1
        param_b.decay_mult = 1
        return [param_w, param_b]
    else:
        raise ValueError("Unknown num_param {}".format(num_param))
    return param
        
def _get_transform_param(tf_spec):
    param = caffe_pb2.TransformationParameter()
    crop_size = tf_spec.get("crop_size")
    if crop_size is not None:
        param.crop_size = crop_size
    mirror = tf_spec.get("mirror")
    if mirror is not None:
        param.mirror = mirror
    else:
        param.mirror = False
    mean_value = tf_spec.get("mean_value")
    if mean_value is not None:
        param.mean_value.extend(mean_value)
    return param
    
def Data(name, tops, source, batch_size, phase):
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'Data'
    layer.top.extend(tops)
    layer.data_param.source = source
    layer.data_param.batch_size = batch_size
    layer.data_param.backend = caffe_pb2.DataParameter.LMDB
    layer.include.extend([_get_include(phase)])
    layer.tranform_param.CopyFrom(_get_transform_param(phase))
    return [layer]
    
def Input(name, tops, shapes, phase):
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'Input'
    layer.top.extend(tops)
    for sp in shapes:
        data_shape = caffe_pb2.BlobShape()
        data_shape.dim.extend(sp)
        layer.input_param.shape.extend([data_shape])
    layer.include.extend([_get_include(phase)])
    return [layer]

def Reshape(name, bottom, shape):
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'Reshape'
    layer.bottom.extend([bottom])
    layer.top.extend([name])
    layer.reshape_param.shape.dim.extend(shape)
    return [layer]

def Conv(name, bottom, num_output, kernel_size, stride, pad, bias_term=False,
         group=1, filler_spec=('msra', 0, 1)):
    """
        filler_spec = (type, min, std)
    """
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'Convolution'
    layer.bottom.extend([bottom])
    layer.top.extend([name])
    layer.convolution_param.num_output = num_output
    layer.convolution_param.kernel_size.extend([kernel_size])
    layer.convolution_param.stride.extend([stride])
    layer.convolution_param.pad.extend([pad])
    if group > 1:
        layer.convolution_param.group = group
    layer.convolution_param.weight_filler.type = filler_spec[0]
    if filler_spec[1] != 0:
        layer.convolution_param.weight_filler.min = filler_spec[1]
    if filler_spec[2] != 1:
        layer.convolution_param.weight_filler.std = filler_spec[2]
    layer.convolution_param.bias_term = bias_term
    if bias_term:
        layer.convolution_param.bias_filler.value = 0
        layer.param.extend(_get_param(2))
    else:
        layer.param.extend(_get_param(1))
    return [layer]

def Deconv(name, bottom, num_output, kernel_size, stride, pad, bias_term=False,
           group=1, filler_spec=('msra', 0, 1)):
    """
        filler_spec = (type, min, std)
    """
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'Deconvolution'
    layer.bottom.extend([bottom])
    layer.top.extend([name])
    layer.convolution_param.num_output = num_output
    layer.convolution_param.kernel_size.extend([kernel_size])
    layer.convolution_param.stride.extend([stride])
    layer.convolution_param.pad.extend([pad])
    if group > 1:
        layer.convolution_param.group = group
    layer.convolution_param.weight_filler.type = filler_spec[0]
    if filler_spec[1] != 0:
        layer.convolution_param.weight_filler.min = filler_spec[1]
    if filler_spec[2] != 1:
        layer.convolution_param.weight_filler.std = filler_spec[2]
    layer.convolution_param.bias_term = bias_term
    if bias_term:
        layer.convolution_param.bias_filler.value = 0
        layer.param.extend(_get_param(2))
    else:
        layer.param.extend(_get_param(1))
    return [layer]

def BatchNorm(name, bottom, moving_average_fraction=0.9, scale_after=True):
    top_name = name
    bn_layer = caffe_pb2.LayerParameter()
    bn_layer.name = name + '_bn'
    bn_layer.type = 'BatchNorm'
    bn_layer.batch_norm_param.moving_average_fraction = moving_average_fraction
    bn_layer.bottom.extend([bottom])
    bn_layer.top.extend([top_name])
    if scale_after is True:
       scale_layer = caffe_pb2.LayerParameter()
       scale_layer.name = name + '_scale'
       scale_layer.type = 'Scale'
       scale_layer.bottom.extend([top_name])
       scale_layer.top.extend([top_name])
       scale_layer.scale_param.filler.value = 1
       scale_layer.scale_param.bias_term = True
       scale_layer.scale_param.bias_filler.value = 0
       return [bn_layer, scale_layer]
    return [bn_layer]
    
def ReLU(name, bottom, negative_slope=0):
    layer = caffe_pb2.LayerParameter()
    layer.name = name + '_relu'
    layer.type = 'ReLU'
    layer.bottom.extend([bottom])
    layer.top.extend([bottom])
    if negative_slope != 0:
        layer.relu_param.negative_slope = negative_slope
    return [layer]

def DuReLU(name, bottom):
    top_name = name
    # positive relu
    pos_relu_layer = caffe_pb2.LayerParameter()
    pos_relu_layer.name = name + '_pos_relu'
    pos_relu_layer.type = 'ReLU'
    pos_relu_layer.bottom.extend([bottom])
    pos_relu_layer.top.extend([top_name+'_pos_relu'])
    # negative
    neg_layer = caffe_pb2.LayerParameter()
    neg_layer.name = name + '_neg'
    neg_layer.type = 'Power'
    neg_layer.bottom.extend([bottom])
    neg_layer.top.extend([top_name + '_neg_relu'])
    neg_layer.power_param.scale = -1
    # negative relu
    neg_relu_layer = caffe_pb2.LayerParameter()
    neg_relu_layer.name = name + '_neg_relu'
    neg_relu_layer.type = 'ReLU'
    neg_relu_layer.bottom.extend([top_name + '_neg_relu'])
    neg_relu_layer.top.extend([top_name + '_neg_relu'])
    # durelu
    durelu_layer = caffe_pb2.LayerParameter()
    durelu_layer.name = name + '_durelu'
    durelu_layer.type = 'Concat'
    durelu_layer.bottom.extend([top_name+'_pos_relu', top_name+'_neg_relu'])
    durelu_layer.top.extend([top_name + '_durelu'])
    return [pos_relu_layer, neg_layer, neg_relu_layer, durelu_layer]

def Act(name, bottom, bn=True, durelu=False, bn_frac=0.9, leaky=0):
    top_name = name
    ret_layers = list()
    if bn:
        # BN
        bn_layer = caffe_pb2.LayerParameter()
        bn_layer.name = name + '_bn'
        bn_layer.type = 'BatchNorm'
        bn_layer.batch_norm_param.moving_average_fraction = bn_frac
        bn_layer.bottom.extend([bottom])
        bn_layer.top.extend([top_name])
        # Scale
        scale_layer = caffe_pb2.LayerParameter()
        scale_layer.name = name + '_scale'
        scale_layer.type = 'Scale'
        scale_layer.bottom.extend([top_name])
        scale_layer.top.extend([top_name])
        scale_layer.scale_param.filler.value = 1
        scale_layer.scale_param.bias_term = True
        scale_layer.scale_param.bias_filler.value = 0
        ret_layers.extend([bn_layer, scale_layer])
    else:
        # Scale
        scale_layer = caffe_pb2.LayerParameter()
        scale_layer.name = name + '_scale'
        scale_layer.type = 'Scale'
        scale_layer.bottom.extend([bottom])
        scale_layer.top.extend([top_name])
        scale_layer.scale_param.filler.value = 1
        scale_layer.scale_param.bias_term = True
        scale_layer.scale_param.bias_filler.value = 0
        ret_layers.extend([scale_layer])
    # Dual relu
    if durelu:
        # positive relu
        pos_relu_layer = caffe_pb2.LayerParameter()
        pos_relu_layer.name = name + '_pos_relu'
        pos_relu_layer.type = 'ReLU'
        pos_relu_layer.bottom.extend([top_name])
        pos_relu_layer.top.extend([top_name + '_pos_relu'])
        # negative
        neg_layer = caffe_pb2.LayerParameter()
        neg_layer.name = name + '_neg'
        neg_layer.type = 'Power'
        neg_layer.bottom.extend([top_name])
        neg_layer.top.extend([top_name + '_neg_relu'])
        neg_layer.power_param.scale = -1
        # negative relu
        neg_relu_layer = caffe_pb2.LayerParameter()
        neg_relu_layer.name = name + '_neg_relu'
        neg_relu_layer.type = 'ReLU'
        neg_relu_layer.bottom.extend([top_name + '_neg_relu'])
        neg_relu_layer.top.extend([top_name + '_neg_relu'])
        # dual relu
        durelu_layer = caffe_pb2.LayerParameter()
        durelu_layer.name = name + '_durelu'
        durelu_layer.type = 'Concat'
        durelu_layer.bottom.extend([top_name+'_pos_relu', top_name+'_neg_relu'])
        durelu_layer.top.extend([top_name+'_durelu'])
        ret_layers.extend([pos_relu_layer, neg_layer, neg_relu_layer, durelu_layer])
    elif leaky == 0:
        # relu
        relu_layer = caffe_pb2.LayerParameter()
        relu_layer.name = name + '_relu'
        relu_layer.type = 'ReLU'
        relu_layer.bottom.extend([top_name])
        relu_layer.top.extend([top_name])
        ret_layers.extend([relu_layer])
    else:
        # leaky relu
        relu_layer = caffe_pb2.LayerParameter()
        relu_layer.name = name + '_leaky_relu'
        relu_layer.type = 'ReLU'
        relu_layer.bottom.extend([top_name])
        relu_layer.top.extend([top_name])
        relu_layer.relu_param.negative_slope = leaky
        ret_layers.extend([relu_layer])
    return ret_layers
    
def Tanh(name, bottom):
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'TanH'
    layer.bottom.extend([bottom])
    layer.top.extend([bottom])
    return [layer]
    
def Sigmoid(name, bottom):
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'Sigmoid'
    layer.bottom.extend([bottom])
    layer.top.extend([bottom])
    return [layer]
    
def Pool(name, bottom, pooling_method, kernel_size, stride, pad):
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'Pooling'
    layer.bottom.extend([bottom])
    layer.top.extend([name])
    if pooling_method == 'max':
        layer.pooling_param.pool = caffe_pb2.PoolingParameter.MAX
    elif pooling_method == 'ave':
        layer.pooling_param.pool = caffe_pb2.PoolingParameter.AVE
    else:
        raise ValueError("Unknown pooling method {}".format(pooling_method))
    layer.pooling_param.kernel_size = kernel_size
    layer.pooling_param.stride = stride
    layer.pooling_param.pad = pad
    return [layer]
    
def Linear(name, bottom, num_output, bias_term=True, filler_spec=('msra', 0, 1)):
    """
        filler_spec = (type, min, std)
    """
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'InnerProduct'
    layer.bottom.extend([bottom])
    layer.top.extend([name])
    layer.inner_product_param.num_output = num_output
    layer.inner_product_param.weight_filler.type = filler_spec[0]
    if filler_spec[1] != 0:
        layer.inner_product_param.weight_filler.min = filler_spec[1]
    if filler_spec[2] != 1:
        layer.inner_product_param.weight_filler.std = filler_spec[2]
    if bias_term:
        layer.inner_product_param.bias_filler.value = 0
        layer.param.extend(_get_param(2))
    else:
        layer.inner_product_param.bias_term = False
        layer.param.extend(_get_param(1))
    return [layer]

def Add(name, bottoms):
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'Eltwise'
    layer.bottom.extend(bottoms)
    layer.top.extend([name])
    return [layer]

def SoftmaxLoss(name, bottoms):
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'SoftmaxWithLoss'
    layer.bottom.extend(bottoms)
    layer.top.extend([name])
    return [layer]
    
def Accuracy(name, bottoms, top_k):
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'Accuracy'
    layer.bottom.extend(bottoms)
    layer.top.extend([name])
    layer.accuracy_param.top_k = top_k
    layer.include.extend([_get_include('test')])
    return [layer]
    
def SumLoss(name, bottom):
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'SumLoss'
    layer.bottom.append(bottom)
    layer.top.append(name)
    return [layer]

def L2Loss(name, bottoms):
    top_name = name
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'EuclideanLoss'
    layer.bottom.extend(bottoms)
    layer.top.extend([top_name])
    return [layer]
    
