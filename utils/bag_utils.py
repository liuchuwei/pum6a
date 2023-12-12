#-*-coding:utf-8 -*-
import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from typing import *
import scipy.io as scio

from utils.gen_data import genData
import random

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

class BagsLoader(data_utils.Dataset):

    r'''
    A torch dataset class for loading bag dataset
    '''

    def __init__(self, bag):
        r'''
        Initialization function for the class
            Args:
                bag: bag dataset
        '''

        self.bags = bag

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, index):

        bag = [self.bags[index], index]
        idx = index

        return bag, idx


class Bags(object):

    r'''
    An object class for generating bag dataset
    '''

    def __init__(self,
                 dataset: Optional[str]="MNIST",
                 num_bag: Optional[int]=250,
                 mean_nbag_length: Optional[int]=10,
                 var_nbag_length: Optional[int]=2,
                 # mean_abag_length: Optional[int]=4,
                 # var_abag_length: Optional[int]=1,
                 confactor: Optional[int]=0.3,
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
        # self.mean_abag_length = mean_abag_length
        # self.var_abag_length = var_abag_length
        self.confactor = confactor
        self.target = target
        self.num_bag = num_bag
        self.r = np.random.RandomState(seed)
        self.obtain_dataset()

        if not dataset in ["MUSK1", "MUSK2", "FOX", "TIGER", "ELEPHANT"]:

            self.bags, self.labels = self.create_bag()

        # self.__getitem__(7)

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

    def load_bag(self, data_file):
        """
        Instance method to load bag dataset from RealWorld
        """
        bags = []
        instances = []
        labels = []
        with open(data_file) as f:
            for l in f.readlines():

                ss = np.array(l.split(',')).astype(np.float32)
                bags.append(str(int(ss[1])))
                labels.append(int(ss[0]))
                instances.append(ss[2:])

        instances = self.zscore_normalize(np.stack(instances))
        old_bag = bags[0]
        label = []
        instance = []
        bag_list = []
        for idx, item in enumerate(bags):
            cur_bag = item
            if cur_bag==old_bag:
                label.append(labels[idx])
                instance.append(instances[idx])
            else:
                old_bag = cur_bag
                bag_list.append({'labels':label, 'instances':instance})
                label = [labels[idx]]
                instance = [instances[idx]]

        last_bag = bags[-1]
        label = []
        instance = []
        for idx, item in enumerate(bags):
            cur_bag = item
            if cur_bag==last_bag:
                label.append(labels[idx])
                instance.append(instances[idx])
        bag_list.append({'labels': label, 'instances': instance})

        return bag_list

    def obtain_dataset(self):

        '''
        Instance method for obtaining dataset
        '''

        if self.dataset == "construct":

            bags, bags_labels, X_inst, y_inst = genData(k=10, nbags=500, bag_contfactor=0.3, seed=331)

            self.n_inst = storeDataset(data=torch.Tensor(X_inst[np.where(y_inst != self.target)]).float(),
                                       target=torch.Tensor(y_inst[np.where(y_inst != self.target)]).float())

            self.a_inst = storeDataset(data=torch.Tensor(X_inst[np.where(y_inst == self.target)]).float(),
                                       target=torch.Tensor(y_inst[np.where(y_inst == self.target)]).float())

        elif self.dataset == "MNIST":

            pipeline = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

            loader = data_utils.DataLoader(datasets.MNIST("dataset",
                                                          train=True,
                                                          download=True,
                                                          transform=pipeline),
                                           batch_size=60000,
                                           shuffle=False)

            for (batch_data, batch_labels) in loader:

                self.n_inst = storeDataset(data=batch_data[torch.where(batch_labels != self.target)],
                                           target=batch_labels[torch.where(batch_labels != self.target)])
                self.a_inst = storeDataset(data=batch_data[torch.where(batch_labels == self.target)],
                                           target=batch_labels[torch.where(batch_labels == self.target)])


        elif self.dataset == "MUSK1":

            data = self.load_bag('dataset/RealWord/Musk1.csv')
            self.labels = [torch.Tensor(item['labels'])==self.target for item in data]
            self.bags = [torch.Tensor(np.array(item['instances'])) for item in data]

        elif self.dataset == "MUSK2":

            data = self.load_bag('dataset/RealWord/Musk2.csv')
            self.labels = [torch.Tensor(item['labels']) == self.target for item in data]
            self.bags = [torch.Tensor(np.array(item['instances'])) for item in data]

        elif self.dataset == "FOX":

            data = self.load_bag('dataset/RealWord/Fox.csv')
            self.labels = [torch.Tensor(item['labels']) == self.target for item in data]
            self.bags = [torch.Tensor(np.array(item['instances'])) for item in data]
            self.bags = [torch.where(torch.isnan(a), torch.full_like(a, 0), a) for a in self.bags]

        elif self.dataset == "TIGER":

            data = self.load_bag('dataset/RealWord/Tiger.csv')
            self.labels = [torch.Tensor(item['labels']) == self.target for item in data]
            self.bags = [torch.Tensor(np.array(item['instances'])) for item in data]

        elif self.dataset == "ELEPHANT":

            data = self.load_bag('dataset/RealWord/Elephant.csv')
            self.labels = [torch.Tensor(item['labels']) == self.target for item in data]
            self.bags = [torch.Tensor(np.array(item['instances'])) for item in data]

        elif self.dataset == "Protein":

            data = self.load_bag('dataset/RealWord/Protein.csv')
            self.labels = [torch.Tensor(item['labels']) == self.target for item in data]
            self.bags = [torch.Tensor(np.array(item['instances'])) for item in data]

        elif self.dataset == "Corel_dogs":

            data = self.load_bag('dataset/RealWord/CorelDogs.csv')
            self.labels = [torch.Tensor(item['labels']) == self.target for item in data]
            self.bags = [torch.Tensor(np.array(item['instances'])) for item in data]

        elif self.dataset == "Ucsb_breast_cancer":

            data = self.load_bag('dataset/RealWord/UCSBBreastCancer.csv')
            self.labels = [torch.Tensor(item['labels']) == self.target for item in data]
            self.bags = [torch.Tensor(np.array(item['instances'])) for item in data]

        elif self.dataset == "Web_recommendation_1":

            data = self.load_bag('dataset/RealWord/Web1.csv')
            self.labels = [torch.Tensor(item['labels']) == self.target for item in data]
            self.bags = [torch.Tensor(np.array(item['instances'])) for item in data]

        elif self.dataset == "Birds_brown_creeper":

            data = self.load_bag('dataset/RealWord/BrownCreeper.csv')
            self.labels = [torch.Tensor(item['labels']) == self.target for item in data]
            self.bags = [torch.Tensor(np.array(item['instances'])) for item in data]

        elif self.dataset == "Annthyroid":

            data = np.load("dataset/ADBench/2_annthyroid.npz", allow_pickle=True)
            X_inst, y_inst = data['X'], data['y']
            X_inst = self.zscore_normalize(X_inst)

            self.n_inst = storeDataset(data=torch.Tensor(X_inst[np.where(y_inst != self.target)]).float(),
                                       target=torch.Tensor(y_inst[np.where(y_inst != self.target)]).float())

            self.a_inst = storeDataset(data=torch.Tensor(X_inst[np.where(y_inst == self.target)]).float(),
                                       target=torch.Tensor(y_inst[np.where(y_inst == self.target)]).float())


        elif self.dataset == "PageBlock":

            data = np.load("dataset/ADBench/27_PageBlocks.npz", allow_pickle=True)
            X_inst, y_inst = data['X'], data['y']
            X_inst = self.zscore_normalize(X_inst)

            self.n_inst = storeDataset(data=torch.Tensor(X_inst[np.where(y_inst != self.target)]).float(),
                                       target=torch.Tensor(y_inst[np.where(y_inst != self.target)]).float())

            self.a_inst = storeDataset(data=torch.Tensor(X_inst[np.where(y_inst == self.target)]).float(),
                                       target=torch.Tensor(y_inst[np.where(y_inst == self.target)]).float())


        elif self.dataset == "SpamBase":

            data = np.load("dataset/ADBench/35_SpamBase.npz", allow_pickle=True)
            X_inst, y_inst = data['X'], data['y']
            X_inst = self.zscore_normalize(X_inst)

            self.n_inst = storeDataset(data=torch.Tensor(X_inst[np.where(y_inst != self.target)]).float(),
                                       target=torch.Tensor(y_inst[np.where(y_inst != self.target)]).float())

            self.a_inst = storeDataset(data=torch.Tensor(X_inst[np.where(y_inst == self.target)]).float(),
                                       target=torch.Tensor(y_inst[np.where(y_inst == self.target)]).float())


        elif self.dataset == "Waveform":

            data = np.load("dataset/ADBench/41_Waveform.npz", allow_pickle=True)
            X_inst, y_inst = data['X'], data['y']
            X_inst = self.zscore_normalize(X_inst)

            self.n_inst = storeDataset(data=torch.Tensor(X_inst[np.where(y_inst != self.target)]).float(),
                                       target=torch.Tensor(y_inst[np.where(y_inst != self.target)]).float())

            self.a_inst = storeDataset(data=torch.Tensor(X_inst[np.where(y_inst == self.target)]).float(),
                                       target=torch.Tensor(y_inst[np.where(y_inst == self.target)]).float())


        elif self.dataset == "Cardio":

            data = np.load("dataset/ADBench/6_cardio.npz", allow_pickle=True)
            X_inst, y_inst = data['X'], data['y']
            X_inst = self.zscore_normalize(X_inst)

            self.n_inst = storeDataset(data=torch.Tensor(X_inst[np.where(y_inst != self.target)]).float(),
                                       target=torch.Tensor(y_inst[np.where(y_inst != self.target)]).float())

            self.a_inst = storeDataset(data=torch.Tensor(X_inst[np.where(y_inst == self.target)]).float(),
                                       target=torch.Tensor(y_inst[np.where(y_inst == self.target)]).float())


        elif self.dataset == "Cardiotoc":

            data = np.load("dataset/ADBench/7_Cardiotocography.npz", allow_pickle=True)
            X_inst, y_inst = data['X'], data['y']
            X_inst = self.zscore_normalize(X_inst)

            self.n_inst = storeDataset(data=torch.Tensor(X_inst[np.where(y_inst != self.target)]).float(),
                                       target=torch.Tensor(y_inst[np.where(y_inst != self.target)]).float())

            self.a_inst = storeDataset(data=torch.Tensor(X_inst[np.where(y_inst == self.target)]).float(),
                                       target=torch.Tensor(y_inst[np.where(y_inst == self.target)]).float())

        elif self.dataset == "Internet":

            data = np.load("dataset/ADBench/17_InternetAds.npz", allow_pickle=True)
            X_inst, y_inst = data['X'], data['y']
            X_inst = self.zscore_normalize(X_inst)

            self.n_inst = storeDataset(data=torch.Tensor(X_inst[np.where(y_inst != self.target)]).float(),
                                       target=torch.Tensor(y_inst[np.where(y_inst != self.target)]).float())

            self.a_inst = storeDataset(data=torch.Tensor(X_inst[np.where(y_inst == self.target)]).float(),
                                       target=torch.Tensor(y_inst[np.where(y_inst == self.target)]).float())

        elif self.dataset == "Mammog":

            data = np.load("dataset/ADBench/23_mammography.npz", allow_pickle=True)
            X_inst, y_inst = data['X'], data['y']
            X_inst = self.zscore_normalize(X_inst)

            self.n_inst = storeDataset(data=torch.Tensor(X_inst[np.where(y_inst != self.target)]).float(),
                                       target=torch.Tensor(y_inst[np.where(y_inst != self.target)]).float())

            self.a_inst = storeDataset(data=torch.Tensor(X_inst[np.where(y_inst == self.target)]).float(),
                                       target=torch.Tensor(y_inst[np.where(y_inst == self.target)]).float())

        elif self.dataset == "Musk":

            data = np.load("dataset/ADBench/25_musk.npz", allow_pickle=True)
            X_inst, y_inst = data['X'], data['y']
            X_inst = self.zscore_normalize(X_inst)

            self.n_inst = storeDataset(data=torch.Tensor(X_inst[np.where(y_inst != self.target)]).float(),
                                       target=torch.Tensor(y_inst[np.where(y_inst != self.target)]).float())

            self.a_inst = storeDataset(data=torch.Tensor(X_inst[np.where(y_inst == self.target)]).float(),
                                       target=torch.Tensor(y_inst[np.where(y_inst == self.target)]).float())

        elif self.dataset == "Optdigits":

            data = np.load("dataset/ADBench/26_optdigits.npz", allow_pickle=True)
            X_inst, y_inst = data['X'], data['y']
            # X_inst = self.zscore_normalize(X_inst)

            self.n_inst = storeDataset(data=torch.Tensor(X_inst[np.where(y_inst != self.target)]).float(),
                                       target=torch.Tensor(y_inst[np.where(y_inst != self.target)]).float())

            self.a_inst = storeDataset(data=torch.Tensor(X_inst[np.where(y_inst == self.target)]).float(),
                                       target=torch.Tensor(y_inst[np.where(y_inst == self.target)]).float())

        elif self.dataset == "Pendigits":

            data = np.load("dataset/ADBench/28_pendigits.npz", allow_pickle=True)
            X_inst, y_inst = data['X'], data['y']
            X_inst = self.zscore_normalize(X_inst)

            self.n_inst = storeDataset(data=torch.Tensor(X_inst[np.where(y_inst != self.target)]).float(),
                                       target=torch.Tensor(y_inst[np.where(y_inst != self.target)]).float())

            self.a_inst = storeDataset(data=torch.Tensor(X_inst[np.where(y_inst == self.target)]).float(),
                                       target=torch.Tensor(y_inst[np.where(y_inst == self.target)]).float())

        elif self.dataset == "Satellite":

            data = np.load("dataset/ADBench/30_satellite.npz", allow_pickle=True)
            X_inst, y_inst = data['X'], data['y']
            X_inst = self.zscore_normalize(X_inst)

            self.n_inst = storeDataset(data=torch.Tensor(X_inst[np.where(y_inst != self.target)]).float(),
                                       target=torch.Tensor(y_inst[np.where(y_inst != self.target)]).float())

            self.a_inst = storeDataset(data=torch.Tensor(X_inst[np.where(y_inst == self.target)]).float(),
                                       target=torch.Tensor(y_inst[np.where(y_inst == self.target)]).float())

        elif self.dataset == "Shuttle":

            data = np.load("dataset/ADBench/32_shuttle.npz", allow_pickle=True)
            X_inst, y_inst = data['X'], data['y']
            X_inst = self.zscore_normalize(X_inst)

            self.n_inst = storeDataset(data=torch.Tensor(X_inst[np.where(y_inst != self.target)]).float(),
                                       target=torch.Tensor(y_inst[np.where(y_inst != self.target)]).float())

            self.a_inst = storeDataset(data=torch.Tensor(X_inst[np.where(y_inst == self.target)]).float(),
                                       target=torch.Tensor(y_inst[np.where(y_inst == self.target)]).float())

        elif self.dataset == "Thyroid":

            data = np.load("dataset/ADBench/38_thyroid.npz", allow_pickle=True)
            X_inst, y_inst = data['X'], data['y']
            X_inst = self.zscore_normalize(X_inst)

            self.n_inst = storeDataset(data=torch.Tensor(X_inst[np.where(y_inst != self.target)]).float(),
                                       target=torch.Tensor(y_inst[np.where(y_inst != self.target)]).float())

            self.a_inst = storeDataset(data=torch.Tensor(X_inst[np.where(y_inst == self.target)]).float(),
                                       target=torch.Tensor(y_inst[np.where(y_inst == self.target)]).float())


        elif self.dataset == "Cover":

            data = np.load("dataset/ADBench/10_cover.npz", allow_pickle=True)
            X_inst, y_inst = data['X'], data['y']
            X_inst = self.zscore_normalize(X_inst)

            self.n_inst = storeDataset(data=torch.Tensor(X_inst[np.where(y_inst != self.target)]).float(),
                                       target=torch.Tensor(y_inst[np.where(y_inst != self.target)]).float())

            self.a_inst = storeDataset(data=torch.Tensor(X_inst[np.where(y_inst == self.target)]).float(),
                                       target=torch.Tensor(y_inst[np.where(y_inst == self.target)]).float())

        elif self.dataset == "Hepatitis":

            data = np.load("dataset/ADBench/15_Hepatitis.npz", allow_pickle=True)
            X_inst, y_inst = data['X'], data['y']
            X_inst = self.zscore_normalize(X_inst)

            self.n_inst = storeDataset(data=torch.Tensor(X_inst[np.where(y_inst != self.target)]).float(),
                                       target=torch.Tensor(y_inst[np.where(y_inst != self.target)]).float())

            self.a_inst = storeDataset(data=torch.Tensor(X_inst[np.where(y_inst == self.target)]).float(),
                                       target=torch.Tensor(y_inst[np.where(y_inst == self.target)]).float())
        elif self.dataset == "Donor":

            data = np.load("dataset/ADBench/11_donors.npz", allow_pickle=True)
            X_inst, y_inst = data['X'], data['y']
            X_inst = self.zscore_normalize(X_inst)

            self.n_inst = storeDataset(data=torch.Tensor(X_inst[np.where(y_inst != self.target)]).float(),
                                       target=torch.Tensor(y_inst[np.where(y_inst != self.target)]).float())

            self.a_inst = storeDataset(data=torch.Tensor(X_inst[np.where(y_inst == self.target)]).float(),
                                       target=torch.Tensor(y_inst[np.where(y_inst == self.target)]).float())
        elif self.dataset == "Backdoor":

            data = np.load("dataset/ADBench/3_backdoor.npz", allow_pickle=True)
            X_inst, y_inst = data['X'], data['y']
            # X_inst = self.zscore_normalize(X_inst)

            self.n_inst = storeDataset(data=torch.Tensor(X_inst[np.where(y_inst != self.target)]).float(),
                                       target=torch.Tensor(y_inst[np.where(y_inst != self.target)]).float())

            self.a_inst = storeDataset(data=torch.Tensor(X_inst[np.where(y_inst == self.target)]).float(),
                                       target=torch.Tensor(y_inst[np.where(y_inst == self.target)]).float())

        elif self.dataset == "Satimage":

            data = np.load("dataset/ADBench/31_satimage-2.npz", allow_pickle=True)
            X_inst, y_inst = data['X'], data['y']
            X_inst = self.zscore_normalize(X_inst)

            self.n_inst = storeDataset(data=torch.Tensor(X_inst[np.where(y_inst != self.target)]).float(),
                                       target=torch.Tensor(y_inst[np.where(y_inst != self.target)]).float())

            self.a_inst = storeDataset(data=torch.Tensor(X_inst[np.where(y_inst == self.target)]).float(),
                                       target=torch.Tensor(y_inst[np.where(y_inst == self.target)]).float())
        elif self.dataset == "Yeast":

            data = np.load("dataset/ADBench/47_yeast.npz", allow_pickle=True)
            X_inst, y_inst = data['X'], data['y']
            X_inst = self.zscore_normalize(X_inst)

            self.n_inst = storeDataset(data=torch.Tensor(X_inst[np.where(y_inst != self.target)]).float(),
                                       target=torch.Tensor(y_inst[np.where(y_inst != self.target)]).float())

            self.a_inst = storeDataset(data=torch.Tensor(X_inst[np.where(y_inst == self.target)]).float(),
                                       target=torch.Tensor(y_inst[np.where(y_inst == self.target)]).float())
        elif self.dataset == "Fraud":

            data = np.load("dataset/ADBench/13_fraud.npz", allow_pickle=True)
            X_inst, y_inst = data['X'], data['y']
            # X_inst = self.zscore_normalize(X_inst)

            self.n_inst = storeDataset(data=torch.Tensor(X_inst[np.where(y_inst != self.target)]).float(),
                                       target=torch.Tensor(y_inst[np.where(y_inst != self.target)]).float())

            self.a_inst = storeDataset(data=torch.Tensor(X_inst[np.where(y_inst == self.target)]).float(),
                                       target=torch.Tensor(y_inst[np.where(y_inst == self.target)]).float())
        elif self.dataset == "WDBC":

            data = np.load("dataset/ADBench/43_WDBC.npz", allow_pickle=True)
            X_inst, y_inst = data['X'], data['y']
            X_inst = self.zscore_normalize(X_inst)

            self.n_inst = storeDataset(data=torch.Tensor(X_inst[np.where(y_inst != self.target)]).float(),
                                       target=torch.Tensor(y_inst[np.where(y_inst != self.target)]).float())

            self.a_inst = storeDataset(data=torch.Tensor(X_inst[np.where(y_inst == self.target)]).float(),
                                       target=torch.Tensor(y_inst[np.where(y_inst == self.target)]).float())

        else:

            raise ValueError('Dataset not support')

    def create_bag(self):

        r'''
        Instance method for generating bag dataset
        '''

        bags_list = []
        labels_list = []
        self.original_label = []

        for i in range(self.num_bag):

            label = np.random.binomial(1, self.confactor, size=1)[0]
            n_bag_length = np.int32(self.r.normal(self.mean_nbag_length, self.var_nbag_length, 1))
            # a_bag_length = np.int32(self.r.normal(self.mean_abag_length, self.var_abag_length, 1))
            a_bag_length = np.int32(random.randint(0, self.mean_nbag_length/2))

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

                if n_bag_length < 1:
                    n_bag_length = 0

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

