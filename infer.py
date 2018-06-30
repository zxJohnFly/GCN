import argparse
import numpy as np
import torch.utils.data as data

from datasets import load_data, db
from torch.autograd import Variable
from models import GCN
from metrics import mAP, topK
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument('n_bits', type=int, default=32)
args = parser.parse_args()

n_anchor = 1000
n_bits =  args.n_bits
n_class = 21
n_epoch = 10
topk = 15
# dataset: 'cifar10', 'nuswide', 'ImageNet', 'sun'
dataset = 'nuswide'
dset = load_data(dataset)

meta = torch.load('nuswide_2000_32_0.4454_0.5912')
anchor = meta['anchor']
gcn = GCN(500, n_bits, n_class, meta['anchor_affnty'], 40)
gcn.load_state_dict(meta['state_dict'])
gcn.cuda()

test_loader = data.DataLoader(dataset=db(dset.testdata, dset.testlabel),
                              batch_size=100,
                              shuffle=False,
                              num_workers=4)
tH = []
gcn.eval()
for images, _ in test_loader:
    in_aff, out_aff = rbf_affnty(images, anchor, topk=topk)
    images = Variable(images).cuda()
    in_aff = Variable(in_aff).cuda()
    out_aff = Variable(out_aff).cuda()

    out, _ = gcn(images, in_aff, out_aff)
    tH.append(out.data.cpu().numpy())
tH = np.sign(np.concatenate(tH))    

train_loader = data.DataLoader(dataset=db(dset.traindata, dset.trainlabel),
                               batch_size=100,
                               shuffle=False,
                               num_workers=4)
H = []
for images, _ in train_loader:
    in_aff, out_aff = rbf_affnty(images, anchor, topk=topk)
    images = Variable(images).cuda()
    in_aff = Variable(in_aff).cuda()
    out_aff = Variable(out_aff).cuda()

    out, _ = gcn(images, in_aff, out_aff)
    H.append(out.data.cpu().numpy())
H = np.sign(np.concatenate(H))    

HammTrainTest = 0.5*(n_bits - np.dot(H, tH.T))
HammingRank = np.argsort(HammTrainTest, axis=0)

ap = mAP(dset.cateTrainTest, HammingRank)
pre, rec = topK(dset.cateTrainTest, HammingRank, 500)

print ('%d bits: mAP: %.4f, precision@500: %.4f, recall@500: %.4F'
       % (n_bits, ap, pre, rec))

import scipy.io as sio
sio.savemat('./%s_%d' % (dataset, n_bits), {'H': H, 'tH': tH})
