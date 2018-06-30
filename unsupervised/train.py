from models import GCN
from utils import *
 
import numpy as np
import scipy.io as sio


output_n = 64 

sample, testdata, cateTrainTest, tradj,  gfunc = load_cifar10(binary=True)
#sample, cateTrainTest, tradj, gfunc1, gfunc2 = load_imagenet(binary=True)
#traindata, cateTrainTest, tradj, gfunc1, gfunc2 = load_nuswide(binary=True)

#traindata, tradj = load_SIFT1M()
trgcn = GCN(sample, output_n, adj=tradj)
trgcn.build(lr=0.5, PCA=True)

for i in range(3):
    for j  in range(100):
        loss = trgcn.train_fn(sample)
     
        print 'epoch {0}/300 loss: {1}'.format(i*300+j+1, loss) 

    nz = trgcn.infer_fn(sample) 
    trgcn.loss_fn.update(nz)   

B = trgcn.infer_fn(sample)
trgcn.save()
#trgcn.load('/mnt/disk1/ImageNett_model_%d' % output_n)
#np.save('%d' % output_n, np.sign(nz))
#after = trgcn.infer_fn(sample)
#tB = np.dot(gfunc1, B)
#sio.savemat('emb', {'B':B})
#B = sio.loadmat('emb')['B']
#trgcn.load('/mnt/disk1/ImageNett_model_%d' % output_n)
B = np.sign(B)
#sio.savemat('/mnt/disk1/ImageNet1M/ImageNettiny/%d.mat' % output_n, {'Z': Z})
#B = np.sign(np.dot(gfunc1, Z))
tB = np.sign(np.dot(gfunc, B))
#ttgcn.build_infer_fn()

#B = np.sign(trgcn.infer_fn(traindata))
#tB = np.sign(ttgcn.infer_fn(testdata))

#np.savez('/mnt/disk1/ImageNet_%d' % output_n, B=B, tB=tB)

HammTrainTest = 0.5*(output_n - np.dot(B, tB.T))
HammingRank = np.argsort(HammTrainTest, axis=0)

ap = mAP(cateTrainTest, HammingRank)
p, r = topK(cateTrainTest, HammingRank, 500)

print 'mAP: %.4f' % ap
print 'precision@500: %.4f, recall@500: %.4f' %(p, r) 


