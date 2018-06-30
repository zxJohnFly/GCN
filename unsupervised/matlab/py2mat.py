import cPickle as pkl
import scipy.io as sio

model = pkl.load(open('/mnt/disk1/ImageNett_model_16')) 
w1, b1, w2, b2, w3, b3 = model

sio.savemat('/mnt/disk1/ImageNett_model_16.mat', {'w1':w1, 'w2':w2, 'w3':w3,
                                               'b1':b1, 'b2':b2, 'b3':b3})
