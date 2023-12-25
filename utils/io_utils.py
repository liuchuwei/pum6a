import torch

from model_factory.PUM6A import pum6a
from model_factory.nano_PUM6a import Nanopum6a
from model_factory.PUMA import puma
from model_factory.IAE import iAE
from model_factory.PUIF import puIF
from model_factory.RandomForest import RF
from typing import *

from trainers.NanoTrainer import nanoTrainer
from trainers.RandomFroestTrainer import RF_Trainer
from utils.bag_utils import Bags
from trainers.AdanTrainer import adanTrainer
from trainers.PUIF_Trainer import puIF_Trainer
from trainers.BaseTrainer import baseTrainer

import numpy as np
import random
from torch.utils.data import Dataset

from collections import defaultdict

#####################################
#
# Load Nanopore Bags
#
#####################################
class nanoBag(Dataset):

    """
    Torch dataset object of nanopore bag dataset
    """
    def __init__(self, config: Dict=None,
                 feature=None,
                 id=None,
                 sitedict: Dict=None,
                 mod=None):

        r'''
        Initialization function for the class

            Args:
                feature : read feature data.
                id : read information.
                sitedict (dict): A dictionary containing site configurations.
                config (dict): A dictionary containing dataset configurations.
                mod: Ground truth of site modification

            Returns:
                None

            Raises:
                ValueError: Raises an exception when some of the listed arguments do not follow the allowed conventions
        '''
        super(nanoBag, self).__init__()

        self.config = config
        self.feature = feature
        self.id = id
        self.mod = mod
        self.sitedict = self.TailorSiteDict(sitedict)
        # self.__getitem__(0)

    def Keymod(self):

        """
        Instance method to obtain key mod
        """

        y_tmp = [item.split("|") for item in self.keys]
        for item in y_tmp:
            del item[-1]
        y_tmp = ["|".join(item) for item in y_tmp]

        self.keys_mod = torch.tensor([item in self.mod for item in y_tmp]).float()

    def TailorSiteDict(self, sitedict):
        """
        Instance method for removing sites with reads less than min_reads
        """

        for key in list(sitedict.keys()):
            item = sitedict[key]
            if len(item) < self.config['min_read']:
                sitedict.pop(key)

        self.keys = list(sitedict.keys())
        if not self.config['inference']:
            self.Keymod()

        return sitedict

    def __getitem__(self, idx: int):
        '''
        Instance method to access features from reads belonging to the idx-th site in data_info attribute
        :param item:
        :return:
        '''

        id = self.keys[idx]

        reads = self.sitedict[id]
        feature = np.stack([self.feature[item, :] for item in reads])

        feature = torch.tensor(feature, dtype=torch.float32)

        return feature, idx

    def __len__(self):
        return len(self.sitedict)

class GenNanoBags(object):

    r'''
    An object class for generating nanopore bag dataset
    '''

    def __init__(self, config: Dict):

        r'''
        Initialization function for the class

            Args:
                config (dict): A dictionary containing dataset configurations.

            Returns:
                None

            Raises:
                ValueError: Raises an exception when some of the listed arguments do not follow the allowed conventions
        '''

        self.config = config
        self.loaddata()

    def extractCurrentAlign(self, item):

        """
        Instance method to load read information (current, alignment, read information)
        """

        # if "AAACA" == motif:
        # id
        id = '|'.join([item[0], item[2], item[8], item[9]])

        # current signal
        cur_mean = [float(item) for item in item[3].split("|")]
        cur_std = [float(item) for item in item[4].split("|")]
        cur_median = [float(item) for item in item[5].split("|")]
        cur_length = [int(item) for item in item[6].split("|")]
        cur = np.stack([cur_mean, cur_std, cur_median, cur_length])

        # matching
        # base, strand, cov, q_mean, q_median, q_std, mis, ins, deli = ele[8].split("|")
        eventalign = item[10].split("|")
        site = eventalign[0] + "|" + eventalign[1]
        mat = ",".join(eventalign[2:]).split(",")
        mat = np.array([float(item) for item in mat])

        return cur, mat, id + "|" + site

    def preprocess(self):

        """
        Instance method to load read information
        """

        "1.loading groundtruth"
        if self.config['inference']:
            self.gt = None

        else:
            fl = self.config['ground_truth']
            ground_truth = []
            for i in open(fl, "r"):

                if i.startswith("#"):
                    continue

                ele = i.rstrip().split()
                ground_truth.append("|".join(ele))


            y_gt = [item.split("|") for item in ground_truth]
            for item in y_gt:
                del item[2]
            self.gt = ["|".join(item) for item in y_gt]

        "2.loading read feature"
        cur_info = []
        mat_info = []
        read_info = []
        path = self.config['signal']
        motif = self.config['motif']

        for i in open(path, "r"):
            ele = i.rstrip().split()

            if ele[2] in motif:

                cur, mat, ids = self.extractCurrentAlign(ele)
                cur_info.append(cur)
                mat_info.append(mat)
                read_info.append(ids)


        return {'read':read_info, 'current':cur_info, 'matching':mat_info}



    def splitData(self):

        """
        Instance method for normalizing 、splitting and building train、 validate and test dataset
        """

        X = np.concatenate([np.stack([np.concatenate(item) for item in self.dl['current']]), np.stack(self.dl['matching'])], axis=1)
        mean_val = np.mean(X, axis=0)
        std_val = np.std(X, axis=0)
        X_N = (X - mean_val) / std_val

        # normalize length and quality
        X[:, 15:25] = X_N[:, 15:25]

        id = np.array([item for item in self.dl['read']])

        # split for train, val and test dataset
        if self.config['inference']:
            self.bag = {"id":id, "feature": X}

        else:
            random.seed(88888888)
            indices = random.sample(range(0, len(id)), len(id))

            valid_size = int(0.2 * len(id))
            train_size = int(0.6 * len(id))

            train_indices = indices[:train_size]
            val_indices = indices[train_size:(train_size+valid_size)]
            test_indices = indices[(train_size+valid_size):]

            self.train = {"id":id[train_indices], "feature": X[train_indices,]}
            self.val = {"id":id[val_indices], "feature": X[val_indices,]}
            self.test = {"id":id[test_indices], "feature": X[test_indices,]}

    def buildNanoBags(self, data: Dict):

        """
        Instance method for building build NanoBags dataset

            Args:
                data(dict): data dictionary containing read feature and read id

            Return:
                bags: NanoBags dataset
        """

        site_dict = defaultdict(dict)
        id = data['id']
        feature = data['feature']
        for index, item in enumerate(id):
            site = "|".join(item.split("|")[4:7]) + "|" + item.split("|")[3]
            if not site_dict[site]:
                site_dict[site] = [index]
            else:
                site_dict[site].append(index)


        bag = nanoBag(feature=feature, sitedict=site_dict, config=self.config, id=id, mod=self.gt)

        return bag


    def loaddata(self):

        print("loading and preprocessing data...")
        self.dl = self.preprocess()
        print("finish!")

        print("normalize and reshape data...")
        self.splitData()

        print("building nanoBags dataset...")
        if self.config['inference']:
            self.Bags = self.buildNanoBags(self.bag)
        else:
            self.trainBags = self.buildNanoBags(self.train)
            self.valBags = self.buildNanoBags(self.val)
            self.testBags = self.buildNanoBags(self.test)

def LoadNanoBags(config: Dict):

    """
    Method to load nanopore dataset and package it into site bag dataset

        Args:
            config (dict): A dictionary containing dataset configurations.

        Return:
            bag: A bag object containing site bag dataset
    """

    bag = GenNanoBags(config)

    return bag


#####################################
#
# Load Experiment Bags
#
#####################################

def LoadBag(config: Dict):

    """
    Method to load dataset and package it into bag dataset

        Args:
            config (dict): A dictionary containing dataset configurations.

        Return:
            bag: A bag object containing bag dataset
    """

    dataset = config['dataset']
    num_bag = config['num_bag']
    mean_nbag_length = config['mean_nbag_length']
    var_nbag_length = config['var_nbag_length']
    # mean_abag_length = config['mean_abag_length']
    # var_abag_length = config['var_abag_length']
    confactor = config['confactor']
    target = config['target']
    seed = config['seed']

    bag = Bags(
        dataset=dataset,
        num_bag=num_bag,
        mean_nbag_length=mean_nbag_length,
        var_nbag_length=var_nbag_length,
        # mean_abag_length=mean_abag_length,
        # var_abag_length=var_abag_length,
        confactor=confactor,
        target=target,
        seed=seed
    )

    bags_num = len(bag.bags)
    num_pos = torch.stack([item.max() for item in bag.labels]).sum()
    feature = bag.bags[0].size()[1]
    print("Bags number: %s; Positive bags number: %s; Feature number: %s"% (bags_num, num_pos.item(), feature))

    return bag


def LoadModel(config):

    """
    Method to load model_factory according to config

        Args:
            config (dict): A dictionary containing dataset configurations.

        Return:
            model: Positive and Unlabeled Multi-Instance Model
    """

    if config['model_chosen'] == 'Nanopum6a':
        model = Nanopum6a(config)

    elif config['model_chosen'] == 'pum6a':
        model = pum6a(config)

    elif config['model_chosen'] == 'puma':
        model = puma(config)

    elif config['model_chosen'] == 'iAE':
        model = iAE(config)

    elif config['model_chosen'] == 'puIF':
        model = puIF(config)

    elif config['model_chosen'] == 'RF':
        model = RF(config)


    return model


def LoadTrainer(config: Dict,
                model,
                bag):

    """
    Method to load model_factory according to config

        Args:
            config (dict): a dictionary containing trainer configurations.
            model (dict): a model to train.
            bag (dict): bag dataset.

        Return:
            trainer: model trainer
    """

    if config['trainer_chosen'] == "adanTrainer":
        trainer = adanTrainer(config=config,
                              model=model,
                              bag=bag)

    elif config['trainer_chosen'] == "puIF_Trainer":
        trainer = puIF_Trainer(config=config,
                              model=model,
                              bag=bag)

    elif config['trainer_chosen'] == "RF_Trainer":
        trainer = RF_Trainer(config=config,
                              model=model,
                              bag=bag)

    elif config['trainer_chosen'] == "Trainer":

        trainer = baseTrainer(config=config,
                             model=model,
                             bag=bag)

    elif config['trainer_chosen'] == "nanoTrainer":

        trainer = nanoTrainer(config=config,
                             model=model,
                             bag=bag)

    return trainer