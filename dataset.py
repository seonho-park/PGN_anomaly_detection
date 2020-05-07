import torch
import torch.utils.data as torchdata
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os


class CIFAR10(torchdata.Dataset):
    def __init__(self, dataroot, target, imagesize, train):
        self.target = target
        self.train = train

        if imagesize is None:
            transform = transforms.Compose([
                transforms.ToTensor(), # first, convert image to PyTorch tensor [0.,1.]
                ])
        else:
            transform = transforms.Compose([
                transforms.Resize(imagesize),
                transforms.ToTensor(), # first, convert image to PyTorch tensor [0.,1.]
                ])
        self.raw_dataset = torchvision.datasets.CIFAR10(root=dataroot, train = train, transform=transform, download = True )
        targets = np.asarray(self.raw_dataset.targets)
        if self.train:
            self.idxs = np.where(targets == target)[0]
        else: # test
            self.idxs = np.arange(targets.size)

    def __len__(self):
        return self.idxs.size

    def __getitem__(self, i):
        idx = self.idxs[i]
        data = self.raw_dataset[idx][0]
        min, max = data.min(), data.max()
        data = (data-min)/(max-min)

        if self.train: # when training, label is not considered
            label = torch.LongTensor([1])
        else: # if label is one, it means it is anomalous
            label = torch.LongTensor([self.raw_dataset.targets[idx]!=self.target])
        return data,label


class MNIST(torchdata.Dataset):
    def __init__(self, dataroot, target, imagesize, train):
        self.target = target
        self.train = train
        if imagesize is None:
            transform = transforms.Compose([
                transforms.ToTensor(), # first, convert image to PyTorch tensor [0.,1.]
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(imagesize),
                transforms.ToTensor(), # first, convert image to PyTorch tensor [0.,1.]
                ])
        self.raw_dataset = torchvision.datasets.MNIST(root=dataroot, train = train, transform=transform, download = True)
        targets = self.raw_dataset.targets.numpy()
        if self.train:
            self.idxs = np.where(targets == target)[0]
        else: # test
            self.idxs = np.arange(targets.size)

        
    def __len__(self):
        return self.idxs.size

    def __getitem__(self, i):
        idx = self.idxs[i]
        data = self.raw_dataset[idx][0]

        if self.train: # when training, label is not considered
            label = torch.LongTensor([1])
        else: # if label is one, it means it is anomalous
            label = torch.LongTensor([self.raw_dataset.targets[idx]!=self.target])
        return data,label


class FMNIST(torchdata.Dataset):
    def __init__(self, dataroot, target, imagesize, train):
        self.target = target
        self.train = train
        if imagesize is None:
            transform = transforms.Compose([
                transforms.ToTensor(), # first, convert image to PyTorch tensor [0.,1.]
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(imagesize),
                transforms.ToTensor(), # first, convert image to PyTorch tensor [0.,1.]
                ])
        self.raw_dataset = torchvision.datasets.FashionMNIST(root=dataroot, train = train, transform=transform, download = True)
        targets = self.raw_dataset.targets.numpy()
        if self.train:
            self.idxs = np.where(targets == target)[0]
        else: # test
            self.idxs = np.arange(targets.size)

        
    def __len__(self):
        return self.idxs.size

    def __getitem__(self, i):
        idx = self.idxs[i]
        data = self.raw_dataset[idx][0]

        if self.train: # when training, label is not considered
            label = torch.LongTensor([1])
        else: # if label is one, it means it is anomalous
            label = torch.LongTensor([self.raw_dataset.targets[idx]!=self.target])
        return data,label


def get_trainloader(datatype, dataroot, target, batchsize, nworkers, imagesize = None):
    if datatype.lower() in ['mnist']:
        traindataset = MNIST(dataroot, target, imagesize = imagesize, train=True)
    elif datatype.lower() in ['cifar10']:
        traindataset = CIFAR10(dataroot, target, imagesize = imagesize, train=True)
    elif datatype.lower() in ['fmnist']:
        traindataset = FMNIST(dataroot, target, imagesize = imagesize, train=True)

    trainloader = torchdata.DataLoader(traindataset, batch_size = batchsize, shuffle = True, num_workers = nworkers)
    return trainloader


def get_testloader(datatype, dataroot, target, batchsize, nworkers, imagesize = None):
    if datatype.lower() in ['mnist']:
        testdataset = MNIST(dataroot, target, imagesize = imagesize, train=False)
    elif datatype.lower() in ['cifar10']:
        testdataset = CIFAR10(dataroot, target, imagesize = imagesize, train=False)
    elif datatype.lower() in ['fmnist']:
        testdataset = FMNIST(dataroot, target, imagesize = imagesize, train=False)

    testloader = torchdata.DataLoader(testdataset, batch_size = batchsize, shuffle = False, num_workers = nworkers)
    return testloader

