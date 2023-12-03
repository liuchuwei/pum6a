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
                 mean_nbag_length: Optional[int]=10,
                 var_nbag_length: Optional[int]=2,
                 mean_abag_length: Optional[int]=4,
                 var_abag_length: Optional[int]=1,
                 confactor: Optional[int]=0.3,
                 n_pos: Optional[int]=50,
                 seed: Optional[int]=8888888,
                 train: Optional[bool]=True):

        r'''
        Initialization function for the class

            Args:
                dataset (str): A dataset used for genrating bag dataset. possible dataset: MNIST、Cifar10、annthyroid
                num_bag: Number of bags to generate
                mean_nbag_length: Mean of normal bags length
                var_nbag_length: Variance of normal bags length
                mean_abag_length: Mean of abnormal bags length
                var_abag_length: Variance of abnormal bags length
                confactor: Ratio of normal bags and abnormal bags
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
        self.mean_nbag_length = mean_nbag_length
        self.var_nbag_length = var_nbag_length
        self.mean_abag_length = mean_abag_length
        self.var_abag_length = var_abag_length
        self.confactor = 0.3
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
            self.adata = storeDataset()
            self.ndata = storeDataset()

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

            self.ndata.data = torch.FloatTensor(self.data.data[torch.where(self.data.targets!=self.target)])
            self.ndata.targets = self.data.targets[torch.where(self.data.targets!=self.target)].float()

            self.adata.data = torch.FloatTensor(self.data.data[torch.where(self.data.targets==self.target)])
            self.adata.targets = self.data.targets[torch.where(self.data.targets==self.target)].float()

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

            self.ndata = storeDataset()
            self.adata = storeDataset()

            self.ndata.data = torch.FloatTensor(X_inst[np.where(y_inst==0)])
            self.ndata.targets = torch.FloatTensor(y_inst[np.where(y_inst==0)])

            self.adata.data = torch.FloatTensor(X_inst[np.where(y_inst==1)])
            self.adata.targets = torch.FloatTensor(y_inst[np.where(y_inst==1)])

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

            label = np.random.binomial(1, self.confactor, size=1)[0]
            n_bag_length = np.int32(self.r.normal(self.mean_nbag_length, self.var_nbag_length, 1))
            a_bag_length = np.int32(self.r.normal(self.mean_abag_length, self.var_abag_length, 1))

            if label == 0:
                bag_length = n_bag_length + a_bag_length

                if bag_length < 1:
                    bag_length = 1

                indices = torch.LongTensor(self.r.randint(0, len(self.ndata.targets), bag_length))

                if not torch.is_tensor(self.ndata.targets):
                    labels_in_bag = torch.tensor(self.ndata.targets)[indices]
                else:
                    labels_in_bag = self.ndata.targets[indices]

                self.original_label.append(labels_in_bag)
                labels_in_bag = labels_in_bag == self.target
                bags_list.append(self.ndata.data[indices])
                labels_list.append(labels_in_bag)

            elif label == 1:

                if a_bag_length < 1:
                    a_bag_length = 1

                n_indices = torch.LongTensor(self.r.randint(0, len(self.ndata.targets), n_bag_length))
                a_indices = torch.LongTensor(self.r.randint(0, len(self.adata.targets), a_bag_length))

                if not torch.is_tensor(self.ndata.targets):
                    labels_in_bag = torch.concat([torch.tensor(self.ndata.targets[n_indices]), torch.tensor(self.adata.targets[a_indices])])
                else:
                    labels_in_bag = torch.concat([self.ndata.targets[n_indices],self.adata.targets[a_indices]])

                self.original_label.append(labels_in_bag)
                labels_in_bag = labels_in_bag == self.target

                if n_bag_length<1:
                    bags_list.append(self.adata.data[a_indices])
                else:
                    bags_list.append(torch.concat([self.ndata.data[n_indices], self.adata.data[a_indices]]))

                labels_list.append(labels_in_bag)


        self.create_bag_label(labels_list)
        if self.train:
            self.pos_bag = [bags_list[item] for item in torch.where(self.bags_labels==1)[0]]
        return bags_list, labels_list

