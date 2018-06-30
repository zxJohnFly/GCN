import argparse
import numpy as np
import torch.utils.data as data
import torch.nn as nn

from itertools import cycle
from datasets import load_ImageNet, load_ImageNet_full, db
from torch.autograd import Variable
from models import GCN
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('n_bits', type=int, default=32)
parser.add_argument('n_labeled', type=int, default=1000)
args = parser.parse_args()

n_labeled = args.n_labeled
n_anchor = 5000
n_bits =  args.n_bits
n_class = 1000
n_epoch = 3 

traindata, label, mean_val = load_ImageNet()
label = label - 1

np.random.seed(1234)
idx = np.arange(len(traindata))
np.random.shuffle(idx)

labeled_data = traindata[idx[:n_labeled], :]
labeled_y = label[0, idx[:n_labeled]]
unlabeled_data = traindata[idx[n_labeled:], :]
unlabeled_y = label[0, idx[n_labeled:]]


labeled_loader = data.DataLoader(dataset=db(labeled_data, labeled_y), #multilabel=True),
                                batch_size=40,
                                shuffle=True,
                                num_workers=0)
labeled_loader = cycle(labeled_loader)

unlabeled_loader = data.DataLoader(dataset=db(unlabeled_data, unlabeled_y), #multilabel=True),
                                  batch_size=80,
                                  shuffle=True,
                                  num_workers=0)

anchor_idx = np.arange(n_labeled)
np.random.shuffle(anchor_idx)
anchor = labeled_data[anchor_idx[:n_anchor], :]
anchor = torch.Tensor(anchor)
anchor_label = labeled_y[anchor_idx[:n_anchor]]

anchor_affnty, _ = labels_affnty(anchor_label, anchor_label)
anchor_affnty = Variable(anchor_affnty).cuda()

gcn = GCN(labeled_data.shape[1], n_bits, n_class, anchor_affnty, 40)
gcn.cuda()

logSoftloss = nn.CrossEntropyLoss()
quan_loss = nn.MSELoss()
#optimizer = torch.optim.SGD(gcn.parameters(), lr=0.001, momentum=0.9)
optimizer = torch.optim.Adam(gcn.parameters(), lr=0.001)

for epoch in range(n_epoch):
    for i, (ux, ul) in enumerate(unlabeled_loader):
        lx, ll = labeled_loader.next()

        images = Variable(torch.cat((lx, ux), 0)).cuda()
        in_aff, out_aff = get_affnty(ll, anchor_label, ux, anchor, topk=4)
        ll = Variable(ll).cuda()

        optimizer.zero_grad()
        outputs, pred = gcn(images, in_aff, out_aff)
        loss = logSoftloss(pred, ll)

        B = torch.sign(outputs).data.cpu().numpy()
        binary_target = Variable(torch.Tensor(B)).cuda()
        loss += quan_loss(outputs, binary_target) * 0.00001
       # loss += torch.norm(torch.dot(outputs.t(), outputs), 2) * 0.00000001
       # loss += torch.sum(outputs)/outputs.size(0) * 0.0000000001
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                   % (epoch+1, n_epoch, i+1, len(unlabeled_loader), loss.data[0]))                   

torch.save({'state_dict':gcn.state_dict(),
            'mean_val': mean_val,             
            'anchor': anchor,
            'anchor_affnty': anchor_affnty}, './ImageNet_%d_%d' % (n_labeled, n_bits))

'''
model = torch.load('./ImageNet_%d_%d' % (n_labeled, n_bits))
print model.keys()
anchor = model['anchor']
anchor_affnty = model['anchor_affnty']
gcn = GCN(4096, n_bits, 1000, anchor_affnty, 40)
gcn.load_state_dict(model['state_dict'])
gcn.cuda()
'''

traindata, testdata = load_ImageNet_full(mean_val)

train_loader = data.DataLoader(dataset=db(traindata, None),
                                       batch_size=100,
                                       shuffle=False,
                                       num_workers=4)
H = []
for images, _ in train_loader:
    in_aff, out_aff = rbf_affnty(images, anchor, topk=20)
    images = Variable(images).cuda()
    in_aff = Variable(in_aff).cuda()
    out_aff = Variable(out_aff).cuda()

    out, _ = gcn(images, in_aff, out_aff)
    H.append(out.data.cpu().numpy())
R = np.concatenate(H)    
H = np.sign(R)    

test_loader = data.DataLoader(dataset=db(testdata, None),
                                       batch_size=100,
                                       shuffle=False,
                                       num_workers=4)
tH = []
for images, _ in test_loader:
    in_aff, out_aff = rbf_affnty(images, anchor, topk=20)
    images = Variable(images).cuda()
    in_aff = Variable(in_aff).cuda()
    out_aff = Variable(out_aff).cuda()

    out, _ = gcn(images, in_aff, out_aff)
    tH.append(out.data.cpu().numpy())
tR = np.concatenate(tH)
tH = np.sign(tR)    

import scipy.io as sio
sio.savemat('ImageNet_%d.mat' % n_bits, {'H':H, 'tH':tH})
