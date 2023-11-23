#-*-coding:utf-8 -*-
import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from typing import *


class Bags(data_utils.Dataset):

    r'''
    A PyTorch Dataset class for generating bag dataset using MNIST dataset or Cifar10 dataset.

    '''

    def __init__(self,
                 dataset: Optional[str]="MNIST",
                 num_bag: Optional[int]=250,
                 mean_bag_length: Optional[int]=10,
                 var_bag_length: Optional[int]=2,
                 seed: Optional[int]=666,
                 train: Optional[bool]=True):

        r'''
        Initialization function for the class

            Args:
                dataset (str): A dataset used for genrating bag dataset. possible dataset: MNIST、Cifar10
                num_bag: Number of bags to generate
                mean_bag_length: Mean of bags length
                var_bag_length: Variance of bags length
                seed: Random seed
                train: whether the dataset is used for training model or testing model

            Returns:
                None

            Raises:
                ValueError: Raises an exception when some of the listed arguments do not follow the allowed conventions
        '''

        if dataset not in ['MNIST', 'Cifar10']:
            raise ValueError("Invalid dataset. possible dataset: MNIST、Cifar10")

        self.dataset = dataset
        self.mean_bag_length = mean_bag_length
        self.var_bag_length = var_bag_length
        self.num_bag = num_bag
        self.r = np.random.RandomState(seed)
        self.train = train

        self.bags, self.labels = self.create_bag()

        # self.__getitem__(0)

    def obtain_dataset(self):

        '''
        Instance method for obtaining dataset
        '''

        if self.dataset == "MNIST":

            '1.MNIST'
            pipeline = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,),(0.3081,))
            ])

            self.target = 9

            if self.train:
                self.data = datasets.MNIST("data",train=True,download=True,transform=pipeline)
            else:
                self.data = datasets.MNIST("data",train=False,download=True,transform=pipeline)

            self.size = self.data.data.size()[0]

        elif self.dataset == "Cifar10":

            '2.Cifar10'

            if self.train:
                self.data = datasets.CIFAR10(root = 'data', train= True, download= True)
            else:
                self.data = datasets.CIFAR10(root = 'data', train= False, download= True)

            self.size = self.data.data.shape[0]
            self.target = 9


    def create_bag(self):

        r'''
        Instance method for generating bag dataset
        '''

        self.obtain_dataset()

        bags_list = []
        labels_list = []

        for i in range(self.num_bag):

            bag_length = np.int32(self.r.normal(self.mean_bag_length, self.var_bag_length, 1))

            if bag_length < 1:
                bag_length = 1

            indices = torch.LongTensor(self.r.randint(0, self.size, bag_length))

            if not torch.is_tensor(self.data.targets):
                labels_in_bag = torch.tensor(self.data.targets)[indices]
            else:
                labels_in_bag = self.data.targets[indices]

            labels_in_bag = labels_in_bag == self.target

            bags_list.append(self.data.data[indices])
            labels_list.append(labels_in_bag)

        return bags_list, labels_list

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        bag = self.bags[index]
        label = [max(self.labels[index]), self.labels[index]]

        return bag, label