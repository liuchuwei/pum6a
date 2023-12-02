#-*-coding:utf-8 -*-
import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from typing import *
from sklearn.model_selection import train_test_split

from utils.gen_data import genData


def inference_collate(batch):

    """
    Assitance method of inference collection for dataloader
    """

    n_instance = [item[0][0].shape[0] for item in batch]
    features = torch.cat([item[0][0] for item in batch])
    bag_idx = [item[0][1] for item in batch]

    return features, n_instance, bag_idx


class storeDataset(object):

    r'''
    An object class for store dataset.

    '''

    def __init__(self):
        r'''
               Initialization function for the class

        '''

        self.data = None
        self.targets = None


class Bags(data_utils.Dataset):

    r'''
    A PyTorch Dataset class for generating bag dataset using MNIST dataset or Cifar10 dataset.

    '''

    def __init__(self,
                 dataset: Optional[str]="MNIST",
                 num_bag: Optional[int]=250,
                 mean_bag_length: Optional[int]=10,
                 var_bag_length: Optional[int]=2,
                 n_pos: Optional[int]=50,
                 seed: Optional[int]=8888888,
                 train: Optional[bool]=True):

        r'''
        Initialization function for the class

            Args:
                dataset (str): A dataset used for genrating bag dataset. possible dataset: MNIST、Cifar10、annthyroid
                num_bag: Number of bags to generate
                mean_bag_length: Mean of bags length
                var_bag_length: Variance of bags length
                n_pos: Number of positive bags
                seed: Random seed
                train: whether the dataset is used for training model or testing model

            Returns:
                None

            Raises:
                ValueError: Raises an exception when some of the listed arguments do not follow the allowed conventions
        '''

        if dataset not in ['MNIST', 'construct', 'annthyroid']:
            raise ValueError("Invalid dataset. possible dataset: MNIST、construct、annthyroid")

        self.dataset = dataset
        self.mean_bag_length = mean_bag_length
        self.var_bag_length = var_bag_length
        self.n_pos = n_pos
        self.num_bag = num_bag
        self.r = np.random.RandomState(seed)
        self.train = train

        self.bags, self.labels = self.create_bag()

        # self.__getitem__(7)

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, index):

        bag = [self.bags[index], index]
        # bag = bag.unsqueeze(0)
        # label = [max(self.labels[index]), self.labels[index]]
        # label = [self.bags_labels[index], self.labels[index]]
        idx = index

        return bag, idx

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

            self.data = storeDataset()

            if self.train:
                loader = data_utils.DataLoader(datasets.MNIST("dataset",train=True,download=True,transform=pipeline),
                                               batch_size=60000,
                                               shuffle=False)
            else:

                loader = data_utils.DataLoader(datasets.MNIST("dataset",train=False,download=True,transform=pipeline),
                                               batch_size=10000,
                                               shuffle=False)

            for (batch_data, batch_labels) in loader:
                self.data.data = batch_data
                self.data.targets = batch_labels

            self.size = self.data.data.size()[0]

        elif self.dataset == "construct":

            '2.construct'

            if self.train:
                bags, bags_labels, X_inst, y_inst = genData(k=10, nbags=500, bag_contfactor=0.3, seed=331)
                X_inst = X_inst[:2500,]
                y_inst = y_inst[:2500,]
            else:
                bags, bags_labels, X_inst, y_inst = genData(k=10, nbags=500, bag_contfactor=0.3, seed=331)
                X_inst = X_inst[2500:,]
                y_inst = y_inst[2500:,]

            mean, std = np.mean(X_inst, axis=0), np.std(X_inst, axis=0)
            X_inst = (X_inst - mean) / std

            self.data = storeDataset()

            self.data.data = torch.FloatTensor(X_inst)
            self.data.targets = torch.FloatTensor(y_inst)

            self.size = self.data.data.shape[0]
            self.target = 1

        elif self.dataset == "annthyroid":

            data = np.load("dataset/ADBench/2_annthyroid.npz", allow_pickle=True)
            X, y = data['X'], data['y']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 42)

            self.data = storeDataset()

            if self.train:
                self.data.data = X_train
                self.data.targets = y_train
            else:
                self.data.data = X_test
                self.data.targets = y_test

            self.size = self.data.data.shape[0]
            self.target = 1

    def create_bag_label(self, labels_list: Optional[list] = None):

        r'''

        Instance method for generating bag pu (positive and unlabeled) labels

            Args:
            labels_list: Instance labels list for generating bag pu labels.

            Returns:
                    None
        '''

        pos = [max(item) for item in labels_list]
        pos_idx = np.random.choice(np.where(pos)[0], size=self.n_pos, replace=False)
        s = np.zeros(len(pos))
        s[pos_idx] = 1

        # neg_idx = np.random.choice(np.where(s == 0)[0], size=self.n_pos, replace=False)
        # s[neg_idx] = -1
        if self.train:
            self.bags_labels = torch.tensor(s)
        else:
            self.bags_labels = torch.stack(pos).float()

    def create_bag(self):

        r'''
        Instance method for generating bag dataset
        '''

        self.obtain_dataset()

        bags_list = []
        labels_list = []
        self.original_label = []
        for i in range(self.num_bag):

            bag_length = np.int32(self.r.normal(self.mean_bag_length, self.var_bag_length, 1))
            # bag_length = np.int32(10)

            if bag_length < 1:
                bag_length = 1

            indices = torch.LongTensor(self.r.randint(0, self.size, bag_length))

            if not torch.is_tensor(self.data.targets):
                labels_in_bag = torch.tensor(self.data.targets)[indices]
            else:
                labels_in_bag = self.data.targets[indices]

            self.original_label.append(labels_in_bag)
            labels_in_bag = labels_in_bag == self.target

            bags_list.append(self.data.data[indices])
            labels_list.append(labels_in_bag)

        # if self.train:

        self.create_bag_label(labels_list)

        return bags_list, labels_list

