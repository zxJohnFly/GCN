import argparse
import time
import numpy as np
import torch.utils.data as data
import torch.nn as nn

from itertools import cycle
from datasets import load_data, db
from torch.autograd import Variable
from models import GCN, GCN_stack
from metrics import mAP, topK
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument('n_bits', type=int, default=32)
parser.add_argument('n_labeled', type=int, default=1000)
parser.add_argument('depth', type=int, default=1)
args = parser.parse_args()

n_labeled = args.n_labeled
n_anchor = 800
n_bits =  args.n_bits
n_class = 10
n_epoch = 14 

n_dim = 2048

# dataset: 'cifar10', 'nuswide', 'ImageNet', 'sun'
dataset = 'cifar10'
dset = load_data(dataset)

# randomly select labeled data
np.random.seed(1234)
idx = np.arange(len(dset.trainlabel))
np.random.shuffle(idx)

labeled_data = dset.traindata[idx[:n_labeled], :]
labeled_y = dset.trainlabel[idx[:n_labeled]]
unlabeled_data = dset.traindata[idx[n_labeled:], :]
unlabeled_y = dset.trainlabel[idx[n_labeled:]]


labeled_loader = data.DataLoader(dataset=db(labeled_data, labeled_y), #multilabel=True),
                                batch_size=80,
                                shuffle=True,
                                num_workers=0)

labeled_loader = cycle(labeled_loader)

unlabeled_loader = data.DataLoader(dataset=db(unlabeled_data, unlabeled_y), #multilabel=True),
                                  batch_size=40,
                                  shuffle=True,
                                  num_workers=0)

anchor_idx = np.arange(n_labeled)
np.random.shuffle(anchor_idx)
anchor = labeled_data[anchor_idx[:n_anchor], :]
anchor = torch.Tensor(anchor)
anchor_label = labeled_y[anchor_idx[:n_anchor]]

anchor_affnty, _ = labels_affnty(anchor_label, anchor_label)
anchor_affnty = Variable(anchor_affnty).cuda()

gcn = GCN_stack(labeled_data.shape[1], n_bits, n_class, anchor_affnty, 60,
                depth=1, n_dim=n_dim)
gcn.cuda()

if dataset.startswith('nuswide'):
    logSoftloss = nn.MultiLabelSoftMarginLoss()
else:    
    logSoftloss = nn.CrossEntropyLoss()
quan_loss = nn.MSELoss()
#aoptimizer = torch.optim.SGD(gcn.parameters(), lr=0.001, momentum=0.9)
optimizer = torch.optim.Adam(gcn.parameters(), lr=0.001)


start = time.clock()
losses = []
for epoch in range(10):
    for i, (ux, ul) in enumerate(unlabeled_loader):
        lx, ll = labeled_loader.next()

        images = Variable(torch.cat((lx, ux), 0)).cuda()
        in_aff, out_aff = get_affnty(ll, anchor_label, ux, anchor, topk=3)
        ll = Variable(ll).cuda()

        optimizer.zero_grad()
        outputs, pred = gcn(images, in_aff, out_aff, lx.size(0))
        loss = logSoftloss(pred, ll)

        B = torch.sign(outputs).data.cpu().numpy()
        binary_target = Variable(torch.Tensor(B)).cuda()
        loss += quan_loss(outputs, binary_target) * 0.00000001
        loss += torch.norm(torch.dot(outputs.t(), outputs), 2) * 0.0000001
        loss += torch.sum(outputs)/outputs.size(0) * 0.0000001
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            pass
          #  print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
          #         % (epoch+1, n_epoch, i+1, len(unlabeled_loader), loss.data[0]))                   

elapsed = (time.clock() - start)
print("Time: ", elapsed)

test_loader = data.DataLoader(dataset=db(dset.testdata, dset.testlabel),
                              batch_size=100,
                              shuffle=False,
                              num_workers=4)
tH = []
gcn.eval()
for images, _ in test_loader:
    in_aff, out_aff = rbf_affnty(images, anchor, topk=20)
    images = Variable(images).cuda()
    in_aff = Variable(in_aff).cuda()
    out_aff = Variable(out_aff).cuda()

    out, _ = gcn(images, in_aff, out_aff, 1)
    tH.append(out.data.cpu().numpy())
tR = np.concatenate(tH)
tH = np.sign(tR)    

train_loader = data.DataLoader(dataset=db(dset.traindata, dset.trainlabel),
                               batch_size=100,
                               shuffle=False,
                               num_workers=4)
H = []
for images, _ in train_loader:
    in_aff, out_aff = rbf_affnty(images, anchor, topk=20)
    images = Variable(images).cuda()
    in_aff = Variable(in_aff).cuda()
    out_aff = Variable(out_aff).cuda()

    out, _ = gcn(images, in_aff, out_aff, 1)
    H.append(out.data.cpu().numpy())
R = np.concatenate(H)    
H = np.sign(R)    

HammTrainTest = 0.5*(n_bits - np.dot(H, tH.T))
HammingRank = np.argsort(HammTrainTest, axis=0)

ap = mAP(dset.cateTrainTest, HammingRank)
pre, rec = topK(dset.cateTrainTest, HammingRank, 500)

print ('%d bits: mAP: %.4f, precision@500: %.4f, recall@500: %.4F'
       % (n_bits, ap, pre, rec))

'''
torch.save({'state_dict':gcn.state_dict(),
            'anchor': anchor,
            'anchor_affnty': anchor_affnty}, './%s_%d_%d_%.4f_%.4f' % (dataset,
            n_labeled, n_bits, ap, pre))

import scipy.io as sio
sio.savemat('./%s_%d_%d' % (dataset, n_labeled, n_bits), {'H': H, 'tH': tH,
                                                          'R': R, 'tR': tR})
                                                          '''
