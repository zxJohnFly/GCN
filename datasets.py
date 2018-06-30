from collections import namedtuple
from PIL import Image 
import torch.utils.data as data
import torch 
import numpy as np
import scipy.io as sio
import h5py as hdf


_paths = {
    'cifar10': '/mnt/disk1/cifar10/cifar10_gist.mat',
    'cifar10_alex': '/mnt/disk1/cifar10_deep/cifar10_alex.mat',
    'cifar10_vgg': '/mnt/disk1/cifar10_deep/cifar10_vgg.mat',
    'sun': '/mnt/disk1/sun_split/sun_split.mat',
    'mnist': '/mnt/disk1/mnist/mnist.mat',
    'nuswide': '/mnt/disk1/NUS_WIDE/NUS_WIDE.mat',
}

dataset = namedtuple('dataset', ['traindata', 'testdata', 
                                 'trainlabel', 'testlabel', 'cateTrainTest'])

def load_nuswide():
    data = sio.loadmat('/mnt/disk1/NUS_WIDE/nuswide_vgg1.mat')
    traindata = np.float32(data['traindata'])
    testdata = np.float32(data['testdata'])
    trainlabel = np.squeeze(np.int32(data['traingnd']))
    testlabel = np.squeeze(np.int32(data['testgnd']))
    cateTrainTest = sio.loadmat('/mnt/disk1/NUS_WIDE/cateTrainTest_vgg.mat')['cateTrainTest']

    traindata = normalize(traindata)
    testdata = normalize(testdata)
    traindata, mean_val = zero_mean(traindata)
    testdata, _ = zero_mean(testdata, mean_val) 

    return dataset(traindata, testdata, trainlabel, testlabel, cateTrainTest)
   
def load_ImageNet():
    d = hdf.File('/mnt/disk1/ImageNet/ImageNet_200k.mat', 'r')
    data = d['/data'].value.T
    label = np.int32(d['/label'].value)

    data = normalize(data)
    data, mean_val = zero_mean(data)
    
    return data, label, mean_val

def load_ImageNet_full(mean_val):
    d = hdf.File('/mnt/disk1/ImageNet/ILSVRC2012_caffe_CNN.mat' , 'r')
    traindata = d['/traindata'].value.T
    testdata = d['/testdata'].value.T

    traindata = normalize(traindata)
    testdata = normalize(testdata)
    traindata, _ = zero_mean(traindata, mean_val) 
    testdata, _ = zero_mean(testdata, mean_val)

    return traindata, testdata

def load_data(dbname):
    if dbname is 'nuswide_alex':
        return load_nuswide()
    assert dbname in _paths, 'Unknown dataset.'
    
    data = sio.loadmat(_paths[dbname])
    traindata = np.float32(data['traindata'])
    testdata = np.float32(data['testdata'])
    trainlabel = np.int32(data['traingnd'])
    testlabel = np.int32(data['testgnd'])
    if dbname is not 'nuswide':
        trainlabel = np.squeeze(trainlabel)
        testlabel = np.squeeze(testlabel)
    cateTrainTest = data['cateTrainTest']

    traindata = normalize(traindata)
    testdata = normalize(testdata)
    traindata, mean_val = zero_mean(traindata)
    testdata, _ = zero_mean(testdata, mean_val) 

    return dataset(traindata, testdata, trainlabel, testlabel, cateTrainTest)

def normalize(x):
    l2_norm = np.linalg.norm(x, axis=1)[:, None]
    l2_norm[np.where(l2_norm == 0)] = 1e-6
    x = x/l2_norm
    return x

def zero_mean(x, mean_val=None):
    if mean_val is None:
        mean_val = np.mean(x, axis=0)
    x -= mean_val
    return x, mean_val


class db(data.Dataset):
    def __init__(self, images, labels, multilabel=False):
        self.images = torch.Tensor(images)
        if labels is None:
            self.labels = torch.zeros(images.shape[0],1)
        else:    
            if multilabel:
                self.labels = torch.Tensor(labels.tolist())
            else:    
                self.labels = torch.LongTensor(labels.tolist())
        self.length = self.images.size(0)

    def __getitem__(self, index):
        return self.images[index, :], self.labels[index]    
        
    def __len__(self):
        return self.length

class ImgFile(data.Dataset):
    def __init__(self, listfile, labels, basedir=None, multilabel=True,
                 transform=None):
        with open(listfile, 'r') as f:
            self.imglist = [l.rstrip().replace('./', basedir) for l in f]
        self.length = len(self.imglist)
        self.transform = transform
        if multilabel:
            self.labels = torch.Tensor(labels.tolist())
        else:
            self.labels = torch.LongTensor(labels.tolist())

    def __getitem__(self, index):     
        img = Image.open(self.imglist[index])
         
        label = self.labels[index]
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, label

    def __len__(self):
        return self.length
