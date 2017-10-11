"""
Most codes from https://github.com/carpedm20/DCGAN-tensorflow
"""
from __future__ import division
import math
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime
from six.moves import xrange
import os, gzip
import _pickle

import tensorflow as tf
import tensorflow.contrib.slim as slim

import matplotlib.pyplot as plt

def unpickle(relpath): 
    print(relpath)  
    with open(relpath, 'rb') as fp:
        d = _pickle.load(fp,encoding='bytes')
    return d

def prepare_input(data=None, labels=None):
    image_height = 32
    image_width = 32
    image_depth = 3
    assert(data.shape[1] == image_height * image_width * image_depth)
    assert(data.shape[0] == labels.shape[0])
    #do mean normaization across all samples
    mu = np.mean(data, axis=0)
    mu = mu.reshape(1,-1)
    sigma = np.std(data, axis=0)
    sigma = sigma.reshape(1, -1)
    data = data - mu
    data = data / sigma
    is_nan = np.isnan(data)
    is_inf = np.isinf(data)
    if np.any(is_nan) or np.any(is_inf):
        print('data is not well-formed : is_nan {n}, is_inf: {i}'.format(n= np.any(is_nan), i=np.any(is_inf)))
    #data is transformed from (no_of_samples, 3072) to (no_of_samples , image_height, image_width, image_depth)
    #make sure the type of the data is no.float32
    data = data.reshape([-1,image_depth, image_height, image_width])
    data = data.transpose([0, 2, 3, 1])
    data = data.astype(np.float32)
    #print("prepare_input: ",len(data),len(labels))
    return data, labels

def read_cifar10(filename): # queue one element

    class CIFAR10Record(object):
        pass
    result = CIFAR10Record()

    label_bytes = 1  # 2 for CIFAR-100
    result.height = 32
    result.width = 32
    result.depth = 3

    data = unpickle(filename)
    #print(data.keys())
    #value = np.asarray(data[b'data']).astype(np.float32)
    #labels = np.asarray(data[b'labels']).astype(np.int32)
    value = np.asarray(data[b'data']).astype(np.float32)
    labels = np.asarray(data[b'labels']).astype(np.int32)
    
    #print("read_cifar10: ",len(value),len(labels))
    return prepare_input(value,labels)
    #return prepare_input(value.astype(np.float32),labels.astype(np.int32))

def load_cifar10():
    data_dir = "\\TensorflowData\\cifar-10-batches-py"

    filenames = [os.path.join(data_dir, 'data_batch_%d' % i) for i in xrange(1, 6)]
    #filenames = ['data_batch_%d.bin' % i for i in xrange(1, 6)]
    filenames.append(os.path.join(data_dir, 'test_batch'))

    for idx , filename in enumerate(filenames):
        temp_X, temp_y = read_cifar10(filename)
        print("load_cifar10 for temp shape:",temp_X.shape,temp_y.shape)
        if idx == 0:
            dataX = temp_X
            labely = temp_y
        else:
            dataX = np.append(dataX,temp_X)
            labely = np.append(labely,temp_y)
        dataX = dataX.reshape([-1,32, 32, 3])
        print("load_cifar10 for len:",len(dataX),len(labely))
        print("load_cifar10 for shape:",dataX.shape,labely.shape)


    seed = 547
    np.random.seed(seed)
    np.random.shuffle(dataX)
    np.random.seed(seed)
    np.random.shuffle(labely)

    y_vec = np.zeros((len(labely), 10), dtype=np.float)
    for i, label in enumerate(labely):
        y_vec[i, labely[i]] = 1.0

    return dataX / 255., y_vec

def load_mnist(dataset_name):
    data_dir = os.path.join("\\TensorflowData\\", dataset_name) # customizing edit

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data

    data = extract_data(data_dir + '\\train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    trX = data.reshape((60000, 28, 28, 1))

    data = extract_data(data_dir + '\\train-labels-idx1-ubyte.gz', 60000, 8, 1)
    trY = data.reshape((60000))

    data = extract_data(data_dir + '\\t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
    teX = data.reshape((10000, 28, 28, 1))

    data = extract_data(data_dir + '\\t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    teY = data.reshape((10000))

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1.0

    return X / 255., y_vec

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def get_image(image_path, input_height, input_width, resize_height=64, resize_width=64, crop=True, grayscale=False):
    image = imread(image_path, grayscale)
    return transform(image, input_height, input_width, resize_height, resize_width, crop)

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path, grayscale = False):
    if (grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c)) # 28 * 8, 28 * 8, c
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1])) # 28 * 8, 28 * 8
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')

def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)

def center_crop(x, crop_h, crop_w, resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def transform(image, input_height, input_width, resize_height=64, resize_width=64, crop=True):
    if crop:
        cropped_image = center_crop(image, input_height, input_width, resize_height, resize_width)
    else:
        cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.


def save_matplot_img(images, size, image_path):
    # revice image data // M*N*3 // RGB float32 : value must set between 0. with 1.
    for idx in range(64):
        vMin = np.amin(images[idx])
        vMax = np.amax(images[idx])
        img_arr = images[idx].reshape(32*32*3,1) # flatten
        for i, v in enumerate(img_arr):
            img_arr[i] = (v-vMin)/(vMax-vMin)
        img_arr = img_arr.reshape(32,32,3) # M*N*3

        # matplot display
        plt.subplot(8,8,idx+1),plt.imshow(img_arr, interpolation='nearest')
        #plt.title("pred.:{}".format(np.argmax(self.data_y[0]),fontsize=10))
        plt.axis("off")

    plt.savefig(image_path)
    #plt.show()            
