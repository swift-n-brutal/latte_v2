# -*- coding: utf-8 -*-
"""
Created on Fri Mar 03 17:15:03 2017

@author: shiwu_001
"""

import os
import os.path as osp
import numpy as np
from scipy import misc, ndimage
DTYPE = np.float32

class ImageDataLoader(object):
    def __init__(self, folder, names, transformer=None, seed=None):
#        super(ImageDataLoader, self).__init__(seed=seed)
        self.rand = np.random.RandomState(seed)
        self.folder = folder
        self.transformer = transformer
        self.names = self._init_name_list(names)
        
    def _init_name_list(self, names):
        fp = file(names, 'r')
        name_list = []
        for line in fp:
            name_list.append(line.strip('\n'))
        self.n_images = len(name_list)
        print self.n_images, 'images in total'
        return name_list
        
    def _load_batch(self, batchids, blob_name, dest):
        for i, index in enumerate(batchids):
            im_path = osp.join(self.folder, self.names[index])
            im = misc.imread(im_path)
            im = im.swapaxes(0,2).swapaxes(1,2)
            if self.transformer is not None:
                im = self.transformer.process(blob_name, im)
            else:
                im = im.astype(DTYPE)
            assert dest[i,...].shape == im.shape, \
                'blob shape is not equal to image shape: {} vs {}'.format(dest[i,...].shape, im.shape)
            dest[i,...] = im[...]
    
    def sample_batch(self, batchsize):
        return self.rand.randint(self.n_images, size=batchsize)
            
    def fill_input(self, blobs, blob_names, batchids):
        assert len(blobs) == 1, 'ImageDataLoader fills only one input blob (%d given)' % len(blobs)
        assert len(blob_names) == 1, 'ImageDataLoader fills only one input blob (%d given)' % len(blob_names)
        blob = blobs[0]
        blob_name = blob_names[0]
        if batchids is None:
            batchids = self.sample_batch(blob.shape[0])
        self._load_batch(batchids, blob_name, blob.data)
        
from multiprocessing import Queue, Process
class ImageDataLoaderPrefetch(ImageDataLoader):
    def __init__(self, queue_size, folder, names, transformer=None, seed=None):
        super(ImageDataLoaderPrefetch, self).__init__(
            folder, names, transformer=transformer, seed=seed)
        # process to sample batchid
        self.batchids_queue_size = queue_size
        self.batchids_queue = None
        self.batchids_process = None
        # processes to load data
        self.data_queue_size = queue_size
        self.blob_names = []
        self.data_queues = {}
        self.data_shapes = {}
        self.worker_processes = []

    def add_prefetch_process(self, blob_name, data_shape):
        batchsize = data_shape[0]
        if self.batchids_process is None:
            self._init_batchids_process(batchsize)
        self.blob_names.append(blob_name)
        self.data_shapes[blob_name] = data_shape
        data_queue = Queue(self.data_queue_size)
        self.data_queues[blob_name] = data_queue
        wp = Process(target=ImageDataLoaderPrefetch._worker_process,
                     args=(blob_name, data_shape, data_queue, self.batchids_queue,
                           self.folder, self.names, self.transformer))
        wp.start()
        self.worker_processes.append(wp)
    
    def _init_batchids_process(self, batchsize):
        if self.batchids_process is not None:
            print 'Batchids process already exists'
            return
        self.batchids_queue = Queue(self.batchids_queue_size)
        self.batchids_process = Process(target=self._batchids_process,
                                        args=(self.rand, self.n_images, batchsize, self.batchids_queue))
        self.batchids_process.start()
    
    @classmethod
    def _batchids_process(cls, rand, n_images, batchsize, batchids_queue):
        while True:
            batchids_queue.put(rand.randint(n_images, size=batchsize))
        
    @classmethod
    def _worker_process(cls, blob_name, data_shape, data_queue, batchids_queue,
                        folder, names, transformer):
        prefetch_data = np.zeros(data_shape, dtype=DTYPE)
        while True:
            batchids = batchids_queue.get()
#            self.dataloader._load_batch(batchids, blob_name, prefetch_data)
            for i, index in enumerate(batchids):
                im_path = osp.join(folder, names[index])
                im = misc.imread(im_path)
                im = im.swapaxes(0,2).swapaxes(1,2)
                if transformer is not None:
                    im = transformer.process(blob_name, im)
                else:
                    im = im.astype(DTYPE)
                assert prefetch_data[i,...].shape == im.shape, 'blob shape is not equal to image shape'
                prefetch_data[i,...] = im[...]
            data_queue.put(prefetch_data)
        
    def _get_data(self, blob_name):
        """
            Return a batch of data for the blob named by blob_name
        """
        dq = self.data_queues.get(blob_name)
        assert dq is not None, 'No such blob specified: %s' % blob_name
        return dq.get()
        
    def clean_and_close(self):
        if self.batchids_process is not None:
            self.batchids_process.terminate()
            self.batchids_process.join()
        for wp in self.worker_processes:
            wp.terminate()
            wp.join()
        if self.batchids_queue is not None:
            self.batchids_queue.close()
            self.batchids_queue.join_thread()
        for name in self.blob_names:
            dq = self.data_queues[name]
            dq.close()
            dq.join_thread()
    
    def fill_input(self, blobs, blob_names, batchids):
        assert len(blobs) == 1, 'ImageDataLoader fills only one input blob (%d given)' % len(blobs)
        assert len(blob_names) == 1, 'ImageDataLoader fills only one input blob (%d given)' % len(blob_names)
        blob = blobs[0]
        blob_name = blob_names[0]
        if batchids is None:
            prefetch_data = self._get_data(blob_name)
            blob.data[...] = prefetch_data
        else:
            self.dataloader._load_batch(batchids, blob_name, blob.data)

class ImageTransformer(object):
    def __init__(self, inputs, seed=None):
#        super(ImageTransformer, self).__init__(seed=seed)
        assert type(inputs) == dict, 'Tranformer.inputs = {blob_name: blob_shape}'
        self.rand = np.random.RandomState(seed)
        self.inputs = inputs
        self.scale = {}
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
            
    def set_scale(self, in_, scale):
        self.__check_input(in_)
        if scale.shape == () or scale.shape == (1,):
            scale = np.array([scale,scale]).reshape(2,1)
        elif scale.shape != (2,) and scale.shape != (2,2):
            raise ValueError('Scale shape invalid. {}'.format(scale.shape))
        self.scale[in_] = scale
            
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
                raise ValueError('Mean shape invalid.')
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
            # elementwise std
            if len(ss) == 2:
                ss = (1,) + ss
            if len(ss) != 3:
                raise ValueError('Std shape invalid.')
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

    def process(self, in_, data):
        self.__check_input(in_)
        data_in = np.copy(data).astype(DTYPE)
        mean = self.mean.get(in_)
        std = self.std.get(in_)
        pad = self.pad.get(in_)
        pad_value = self.pad_value.get(in_)
        scale = self.scale.get(in_)
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
        if scale is not None:
            if scale.shape == (2,2):
                # rand scale, individual ratios
                randsh = self.rand.rand()
                randsw = self.rand.rand()
                scaleh = scale[0,0]*(1-randsh) + scale[0,1]*randsh
                scalew = scale[1,0]*(1-randsw) + scale[0,1]*randsh
            elif scale.shape == (2,):
                # rand scale, keep the ratio of h and w
                randsc = self.rand.rand()
                scaleh = scale[0]*(1-randsc) + scale[1]*randsc
                scalew = scale[0]*(1-randsc) + scale[1]*randsc
            elif scale.shape == (2,1):
                # fixed scale
                scaleh = scale[0]
                scalew = scale[1]
            else:
                scaleh = 1.0
                scalew = 1.0
            # bilinear interpolation
            data_in = ndimage.zoom(data_in, (1.0, scaleh, scalew), order=1)
            
        if data_in.shape[1] >= in_dims[0] and data_in.shape[2] >= in_dims[1]:
            if center is not None and center:
                h_off = int((data_in.shape[1] - in_dims[0] + 1)/2)
                w_off = int((data_in.shape[2] - in_dims[1] + 1)/2)
            else:
                h_off = self.rand.randint(data_in.shape[1] - in_dims[0] + 1)
                w_off = self.rand.randint(data_in.shape[2] - in_dims[1] + 1)
            data_in = data_in[:, h_off:h_off+in_dims[0],
                              w_off:w_off+in_dims[1]]
        else:
            raise ValueError('Image is smaller than input: {} vs {}'.format(
                             data_in.shape[1:], in_dims))
        if mirror is not None and mirror and self.rand.randint(2) == 1:
            data_in = data_in[:,:,::-1]
        return data_in
    
    def deprocess(self, in_, data):
        self.__check_input(in_)
        data_in = np.copy(data).astype(DTYPE)
        std = self.std.get(in_)
        mean = self.mean.get(in_)
        if std is not None:
            data_in *= std
        if mean is not None:
            data_in += mean
        return data_in
    