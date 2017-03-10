# -*- coding: utf-8 -*-
"""
Created on Wed Nov 09 17:25:55 2016

@author: shiwu_001
"""
from .config import DTYPE
import caffe
import numpy as np
import lmdb
from threading import Thread
from Queue import Queue

class DataLoader(object):
    def __init__(self):
        self.mean = np.zeros(3)
        self.std = np.ones(3)
        self.nimages = 0
        self.key_length = 5
        self.data = None
        self.label = None
        self.transformer = None
    
    def __getitem__(self, key):
        if key in ['data', 'label', 'transformer', 'mean', 'std']:
            return eval('self.%s' % key)
        else:
            raise Exception('Invalid key: %s' % key)
            return None
    
    def __setitem__(self, key, item):
        if key in ['data', 'label', 'transformer', 'mean', 'std']:
            exec('self.%s = item' % key)
        else:
            raise Exception('Invalid key: %s' % key)
        
    def _init_loader(self, path):
        env = lmdb.open(path, readonly=True)
        self.nimages = env.stat()['entries']
        print "Load %d images from %s" % (self.nimages, path)
        txn = env.begin()
        return txn
        
    def _load_image(self, txn, index, key_length):
        raw_datum = txn.get(eval("'%%0%dd' %% index" % key_length))
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(raw_datum)
        flat_x = np.fromstring(datum.data, dtype=np.uint8)
        x = flat_x.reshape(datum.channels, datum.height, datum.width)
        y = datum.label
        return x,y
    
    def _load_batch(self, txn, batch, dest_x, dest_y):
        for i,index in enumerate(batch):
            x,y = self._load_image(txn, index, self.key_length)
            dest_x[i,...] = x
            dest_y[i] = y

    def compute_meanstd(data, verbose=False):
        print "Computing mean"
        mean = np.mean(data, axis=(0,2,3))
        print "Computing std"
        std = np.std(data, axis=(0,2,3))
        return mean, std
    
    def load_dataset(self, path, recompute=False, transform=False):
        self.path = path
        txn = self._init_loader(path)
        # get the shape
        x,y = self._load_image(txn, 0, self.key_length)
        c,h,w = x.shape
        data = np.zeros((self.nimages, c, h, w))
        label = np.zeros((self.nimages))
        batch = np.arange(self.nimages)
        self._load_batch(txn, batch, data, label)
        self.data = data
        self.label = label
        if recompute:
            self.mean, self.std = self.compute_meanstd(self.data, verbose=True)
            print "Mean", self.mean
            print "Std", self.std
        if transform:
            self.transform_dataset(self)
    
    def transform_dataset(self, dataset, meanstd=None):
        if meanstd is None:
            dataset['data'] -= self.mean.reshape(1,3,1,1)
            dataset['data'] /= self.std.reshape(1,3,1,1)
        else:
            dataset['data'] -= meanstd['mean']
            dataset['data'] /= meanstd['std']

class CifarDataLoader(DataLoader):
    def __init__(self, path, net, phase, data_blob='data', label_blob='label'):
        super(CifarDataLoader, self).__init__()
        self.mean = np.array([125.3, 123.0, 113.9])
        self.std = np.array([63.0, 62.1, 66.7])
        self.key_length = 5
        self.data_blob = data_blob
        self.label_blob = label_blob
#        self.batchsize = net.blobs[data_blob].num
        if phase == caffe.TRAIN:
            self.load_dataset(path=path, transform=True)
            self.transformer = CifarTransformer({
                data_blob: net.blobs[data_blob].data.shape})
            self.transformer.set_pad(data_blob, 4)
            self.transformer.set_mirror(data_blob, True)
        elif phase == caffe.TEST:
            self.load_dataset(path=path, transform=True)
        else:
            raise Exception("Invalid phase: %s" % str(phase))
        
    def _load_batch_from_dataset(self, batchid, dest_data, dest_label):
        if self.transformer is None:
            for i,bid in enumerate(batchid):
                dest_data[i,...] = self.data[bid,...]
        else:
            for i,bid in enumerate(batchid):
                dest_data[i,...] = self.transformer.process(
                    self.data_blob, self.data[bid,...])
        if dest_label is not None:
            for i,bid in enumerate(batchid):
                dest_label[i] = self.label[bid]
        
    def sample_batch(self, batchsize):
        return np.random.randint(self.nimages, size=batchsize)
    
    def fill_input(self, net, batchid=None):
        data_blob = self.data_blob
        label_blob = self.label_blob
        batchsize = net.blobs[self.data_blob].num
        if batchid is None:
            batchid = self.sample_batch(batchsize)
        else:
            assert(batchsize == len(batchid))
        if label_blob is not None:
            self._load_batch_from_dataset(batchid, net.blobs[data_blob].data,
                                          net.blobs[label_blob].data)
        else:
            self._load_batch_from_dataset(batchid, net.blobs[data_blob].data,
                                          None)

class CifarTransformer(object):
    def __init__(self, inputs):
        self.inputs = inputs
        self.pad = {}
        self.pad_value = {}
        self.mean = {}
        self.std = {}
        self.mirror = {}
        self.center = {}
        
    def __check_input(self, in_):
        if in_ not in self.inputs:
            raise Exception("{} is not one of the net inputs: {}".format(
                in_, self.inputs))
    
    def process(self, in_, data):
        self.__check_input(in_)
        data_in = np.copy(data).astype(DTYPE)
        mean = self.mean.get(in_)
        std = self.std.get(in_)
        pad = self.pad.get(in_)
        pad_value = self.pad_value.get(in_)
        mirror = self.mirror.get(in_)
        center = self.center.get(in_)
        in_dims = self.inputs[in_][2:]
        if mean is not None:
            data_in -= mean
        if std is not None:
            data_in /= std
        if pad is not None:
            if pad_value is None:
                pad_value = 0
            data_in = np.pad(data_in, ((0,0), (pad,pad), (pad,pad)),
                             'constant', constant_values=pad_value)
        if data_in.shape[1] >= in_dims[0] and data_in.shape[2] >= in_dims[1]:
            if center is not None and center:
                h_off = int((data_in.shape[1] - in_dims[0]+1) / 2)
                w_off = int((data_in.shape[2] - in_dims[1]+1) / 2)
            else:
                h_off = np.random.randint(data_in.shape[1] - in_dims[0]+1)
                w_off = np.random.randint(data_in.shape[2] - in_dims[1]+1)
            data_in = data_in[:,h_off:h_off+in_dims[0],
                              w_off:w_off+in_dims[1]]
        else:
            print 'Image is smaller than input: (%d,%d) vs (%d,%d)' \
                % (data_in.shape[1],data_in.shape[2], in_dims[0],in_dims[1])
        if mirror is not None and mirror and np.random.randint(2) == 1:
            data_in = data_in[:,:,::-1]
        return data_in
    
    def set_mean(self, in_, mean):
        self.__check_input(in_)
        ms = mean.shape
        if mean.ndim == 1:
            # broadcast channels
            if ms[0] != self.inputs[in_][1]:
                raise ValueError('Mean channels incompatible with input.')
            mean = mean[:, np.newaxis, np.newaxis]
        else:
            # elementwise mean
            if len(ms) == 2:
                ms = (1,) + ms
            if len(ms) != 3:
                raise ValueError('Mean shape invalid')
            if ms != self.inputs[in_][1:]:
                raise ValueError('Mean shape incompatible with input shape.')
        self.mean[in_] = mean

    def set_std(self, in_, std):
        self.__check_input(in_)
        ss = std.shape
        if std.ndim == 1:
            # broadcast channels
            if ss[0] != self.inputs[in_][1]:
                raise ValueError('Std channels incompatible with input.')
            std = std[:, np.newaxis, np.newaxis]
        else:
            # elementwise mean
            if len(ss) == 2:
                ss = (1,) + ss
            if len(ss) != 3:
                raise ValueError('Std shape invalid')
            if ss != self.inputs[in_][1:]:
                raise ValueError('Std shape incompatible with input shape.')
        self.std[in_] = std
        
    def set_pad(self, in_, pad):
        self.__check_input(in_)
        self.pad[in_] = pad
        
    def set_pad_value(self, in_, pad_value):
        self.__check_input(in_)
        self.pad_value[in_] = pad_value
    
    def set_mirror(self, in_, mirror):
        self.__check_input(in_)
        self.mirror[in_] = mirror

    def set_center(self, in_, center):
        self.__check_input(in_)
        self.center[in_] = center      

class CifarDataLoaderThread(Thread):
    def __init__(self, tid, queue, buffer_out, data, label,
                 data_blob, label_blob, data_blob_shape, transformer):
        super(CifarDataLoaderThread, self).__init__()
        self.tid = tid
        self.queue = queue
        self.buffer_out = buffer_out
        self.data = data
        self.label = label
        self.data_blob = data_blob
        self.label_blob = label_blob
        self.data_blob_shape = data_blob_shape
        self.transformer = transformer
        
    def run(self):
        if self.transformer is not None:
            ndata = self.data.shape[0]
            data_processed = np.zeros(self.data_blob_shape)
            for i in xrange(ndata):
                data_processed[i, ...] = self.transformer.process(self.data_blob, self.data[i, ...])
            self.buffer_out[self.data_blob][...] = data_processed
        else:
            self.buffer_out[self.data_blob][...] = self.data
        if self.label_blob is not None:
            self.buffer_out[self.label_blob][...] = self.label
        self.queue.put(self.tid)

class CifarDataLoaderMultiThreading(CifarDataLoader):
    def __init__(self, nthreads, *args, **kwargs):
        super(CifarDataLoaderMultiThreading, self).__init__(*args, **kwargs)
        self.nthreads = nthreads
        self.thread_list = None
        self.buffers = None
        self.queue = Queue()
        
    def _start_load_batch_from_dataset(self, tid, data_blob_shape):
        batchid = self.sample_batch(data_blob_shape[0])
        td = CifarDataLoaderThread(tid, self.queue, self.buffers[tid],
                                   self.data[batchid, ...], self.label[batchid],
                                   self.data_blob, self.label_blob,
                                   data_blob_shape, self.transformer)
        td.start()
        self.thread_list[tid] = td
        
    def _join_load_batch_from_dataset(self, tid):
        self.thread_list[tid].join()
    
    def _init_load_batch_from_dataset(self, net):
        self.thread_list = range(self.nthreads)
        self.buffers = []
        for tid in xrange(self.nthreads):
            self.buffers.append(dict())
            self.buffers[-1][self.data_blob] = np.zeros_like(net.blobs[self.data_blob].data)
            if self.label_blob is not None:
                self.buffers[-1][self.label_blob] = np.zeros_like(net.blobs[self.label_blob].data)
        for tid in xrange(self.nthreads):
            self._start_load_batch_from_dataset(tid, net.blobs[self.data_blob].data.shape)

    def _fill_input_from_buffers(self, net):
        if self.thread_list is None:
            self._init_load_batch_from_dataset(net)
            print "Init %d dataloader threads" % self.nthreads
        tid = self.queue.get()
        self._join_load_batch_from_dataset(tid)
        net.blobs[self.data_blob].data[...] = self.buffers[tid][self.data_blob]
        if self.label_blob is not None:
            net.blobs[self.label_blob].data[...] = self.buffers[tid][self.label_blob]
        self._start_load_batch_from_dataset(tid, net.blobs[self.data_blob].data.shape)
    
    def fill_input(self, net, batchid=None):
        if batchid is None:
            self._fill_input_from_buffers(net)
        else:
            if self.label_blob is not None:
                self._load_batch_from_dataset(batchid, net.blobs[self.data_blob].data,
                                              net.blobs[self.label_blob].data)
            else:
                self._load_batch_from_dataset(batchid, net.blobs[self.data_blob].data,
                                              None)
    
#    def __del__(self):
#        if self.thread_list is not None:
#            for tid in xrange(self.nthreads):
#                if type(self.thread_list[tid]) is CifarDataLoaderThread:
#                    self.thread_list[tid].terminate()

#if __name__ == '__main__':
#    import sys
#    sys.path.insert(0, '..')
#    from latte.config import CAFFE_ROOT
#    from latte.net import MyNet
#    import os.path as osp
#    import sys
#    import time
#    nthrd = int(sys.argv[1])
#    deploy = osp.join(CAFFE_ROOT, "examples", "cifar10", "resnet20_cifar10_1st_deploy.prototxt")
#    model = None
#    data_blob = "data"
#    label_blob = "label"
#    net = MyNet(deploy, model, pretrained=(model!=None))
#    dataset = CifarDataLoaderMultiThreading(nthrd, osp.join(CAFFE_ROOT, 'examples', 'cifar10/cifar10_train_lmdb'),
#                                    net, phase=caffe.TRAIN,
#                                    data_blob=data_blob, label_blob=label_blob)
#    ######################################################
##    dataset.transformer.set_mirror(data_blob, False)
#    ######################################################
#    net.set_dataloader(dataset)
#    start_time = time.time()
#    for i in xrange(51):
#        net.load_data(None)
#        time.sleep(1)
#        if i % 10 == 0:
#            end_time = time.time()
#            print i, "%.3f" % (end_time - start_time)
#            start_time = end_time