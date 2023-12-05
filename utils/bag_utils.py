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

    def __init__(self, data, target):
        r'''
        Initialization function for the class
            Args:
                  data (torch.Tensor): instance feature
                  target (torch.Tensor): instance target
        '''

        self.X = data
        self.y = target


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
                 target: Optional[int]=1,
                 seed: Optional[int]=8888888
                 ):

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
                target: Which target is defined as abnormal
                seed: Random seed

            Returns:
                None

            Raises:
                ValueError: Raises an exception when some of the listed arguments do not follow the allowed conventions
        '''

        self.dataset = dataset
        self.mean_nbag_length = mean_nbag_length
        self.var_nbag_length = var_nbag_length
        self.mean_abag_length = mean_abag_length
        self.var_abag_length = var_abag_length
        self.confactor = confactor
        self.n_pos = n_pos
        self.target = target
        self.num_bag = num_bag
        self.r = np.random.RandomState(seed)
        self.bags, self.labels = self.create_bag()

        # self.__getitem__(7)

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, index):

        bag = [self.bags[index], index]
        idx = index

        return bag, idx

    def zscore_normalize(self, x):

        r"""
        Instance method to preform z_score normalization

            Args:
                x (numpy.array): numpy array for z_score normalization

            Return:

               X_inst (numpy.array): numpy array after z_score normalization
        """

        mean, std = np.mean(x, axis=0), np.std(x, axis=0)
        X_inst = (x - mean) / std

        return X_inst


    def obtain_dataset(self):

        '''
        Instance method for obtaining dataset
        '''

        if self.dataset == "construct":

            bags, bags_labels, X_inst, y_inst = genData(k=10, nbags=500, bag_contfactor=0.3, seed=331)

            self.n_inst = storeDataset(data=torch.Tensor(X_inst[np.where(y_inst == self.target)]).float(),
                                       target=torch.Tensor(y_inst[np.where(y_inst == self.target)]).float())

            self.a_inst = storeDataset(data=torch.Tensor(X_inst[np.where(y_inst != self.target)]).float(),
                                       target=torch.Tensor(y_inst[np.where(y_inst != self.target)]).float())

        elif self.dataset == "MNIST":

            bag = Bags(dataset="MNIST")

        elif self.dataset == "MUSK1":

            bag = Bags(dataset="MUSK1")

        elif self.dataset == "MUSK2":

            bag = Bags(dataset="MUSK2")

        elif self.dataset == "FOX":

            bag = Bags(dataset="FOX")

        elif self.dataset == "TIGER":

            bag = Bags(dataset="TIGER")

        elif self.dataset == "ELEPHANT":

            bag = Bags(dataset="ELEPHANT")

        elif self.dataset == "Annthyroid":

            bag = Bags(dataset="Annthyroid")

        elif self.dataset == "PageBlock":

            bag = Bags(dataset="PageBlock")

        elif self.dataset == "SpamBase":

            bag = Bags(dataset="SpamBase")

        elif self.dataset == "Waveform":

            bag = Bags(dataset="Waveform")

        elif self.dataset == "Cardio":

            bag = Bags(dataset="Cardio")

        elif self.dataset == "Cardiotoc":

            bag = Bags(dataset="Cardiotoc")

        elif self.dataset == "Internet":

            bag = Bags(dataset="Internet")

        elif self.dataset == "Landsat":

            bag = Bags(dataset="Landsat")

        elif self.dataset == "Letter":

            bag = Bags(dataset="Letter")

        elif self.dataset == "Mammog":

            bag = Bags(dataset="Mammog")

        elif self.dataset == "Musk":

            bag = Bags(dataset="Musk")

        elif self.dataset == "Optdigits":

            bag = Bags(dataset="Optdigits")

        elif self.dataset == "Pendigits":

            bag = Bags(dataset="Pendigits")

        elif self.dataset == "Satellite":

            bag = Bags(dataset="Satellite")

        elif self.dataset == "Shuttle":

            bag = Bags(dataset="Shuttle")

        elif self.dataset == "Skin":

            bag = Bags(dataset="Skin")

        elif self.dataset == "Pima":

            bag = Bags(dataset="Pima")

        elif self.dataset == "Thyroid":

            bag = Bags(dataset="Thyroid")

        elif self.dataset == "Vowels":

            bag = Bags(dataset="Vowels")

        elif self.dataset == "Wilt":

            bag = Bags(dataset="Wilt")

        else:

            raise ValueError('Dataset not support')

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

                indices = torch.LongTensor(self.r.randint(0, len(self.n_inst.y), bag_length))


                labels_in_bag = self.n_inst.y[indices]

                self.original_label.append(labels_in_bag)
                labels_in_bag = labels_in_bag == self.target
                bags_list.append(self.n_inst.X[indices])
                labels_list.append(labels_in_bag)

            elif label == 1:

                if a_bag_length < 1:
                    a_bag_length = 1

                n_indices = torch.LongTensor(self.r.randint(0, len(self.n_inst.y), n_bag_length))
                a_indices = torch.LongTensor(self.r.randint(0, len(self.a_inst.y), a_bag_length))


                labels_in_bag = torch.concat([self.n_inst.y[n_indices], self.a_inst.y[a_indices]])

                self.original_label.append(labels_in_bag)
                labels_in_bag = labels_in_bag == self.target

                if n_bag_length < 1:
                    bags_list.append(self.a_inst.X[a_indices])
                else:
                    bags_list.append(torch.concat([self.n_inst.X[n_indices], self.a_inst.X[a_indices]]))

                labels_list.append(labels_in_bag)

        return bags_list, labels_list

