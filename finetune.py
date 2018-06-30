import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torch.utils.data as data
import numpy as np
import scipy.io as sio

from datasets import ImgFile
from torch.autograd import Variable

epochs = 0 
net = models.vgg.vgg19(pretrained=True)
classifier = list(net.classifier.children())
classifier[-1] = nn.Linear(4096, 21) 
net.classifier = nn.Sequential(*classifier)

train_transform = transforms.Compose([
#    transforms.Scale(256, interpolation=2),
    transforms.RandomSizedCrop(224),    
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Scale(256, interpolation=2),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#traindata = dset.CIFAR10('/mnt/disk1/cifar10/', train=True, transform=transform)
#testdata = dset.CIFAR10('/mnt/disk1/cifar10/', train=False, transform=transform)

basedir = '/mnt/disk1/NUS_WIDE/'
meta = sio.loadmat(basedir+'meta.mat')
traindata = ImgFile(basedir+'train_list.txt', labels=meta['Y_train'],
                    basedir=basedir, transform=train_transform)
testdata = ImgFile(basedir+'test_list.txt', labels=meta['Y_test'],
                   basedir=basedir, transform=val_transform)

train_loader = data.DataLoader(dataset=traindata, batch_size=100, shuffle=True)
test_loader = data.DataLoader(dataset=testdata, batch_size=100, shuffle=False)

#net = nn.DataParallel(net.features)
net.cuda()
#criterion = nn.CrossEntropyLoss()
criterion = nn.MultiLabelSoftMarginLoss()
lr = 0.001
optimizer_ft = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)

for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        images, labels = Variable(images).cuda(), Variable(labels).cuda()
        optimizer_ft.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_ft.step()

        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Iter [%d/%d], Loss: %.4f'
                 % (epoch+1, epochs, i+1, len(train_loader), loss.data[0]))

    total = 0.0
    correct = 0.0
    for images, labels in test_loader:
        images = Variable(images).cuda()
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        predicted = torch.clamp(torch.round(outputs.data), 0, 1).cpu().numpy()
        total += labels.size(0)
        correct += (predicted == labels.numpy()).mean().sum()

    print ('Epoch [%d/%d] acc: %.4f' % (epoch+1, epochs, 100*correct/total))    

    if epoch == 7 or epoch == 14:
        lr *= 0.1
        optimizer_ft = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)

extractor = nn.Sequential(*(list(net.classifier.children())[:-1])) 
net.classifier = extractor
net.eval()

traindata = ImgFile(basedir+'train_list.txt', labels=meta['Y_train'],
                    basedir=basedir, transform=val_transform)
testdata = ImgFile(basedir+'test_list.txt', labels=meta['Y_test'],
                   basedir=basedir, transform=val_transform)

train_loader = data.DataLoader(dataset=traindata, batch_size=50, shuffle=False)
test_loader = data.DataLoader(dataset=testdata, batch_size=50, shuffle=False)


traindata = []
traingnd = []
testdata = []
testgnd = []
for images, labels in train_loader:
    images = Variable(images).cuda()
    feats = net(images)

    traindata.append(feats.data.cpu().numpy())
    traingnd.append(labels.cpu().numpy())

for images, labels in test_loader:
    images = Variable(images).cuda()
    feats = net(images)

    testdata.append(feats.data.cpu().numpy())
    testgnd.append(labels.cpu().numpy())

traindata = np.concatenate(traindata)
testdata = np.concatenate(testdata)
traingnd = np.concatenate(traingnd)
testgnd = np.concatenate(testgnd)

sio.savemat('/mnt/disk1/NUS_WIDE/nuswide_vgg1.mat', {'traindata': traindata,
                                     'testdata': testdata,
                                     'traingnd': traingnd,
                                     'testgnd': testgnd})
