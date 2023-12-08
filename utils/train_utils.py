import os
import string
import random
import numpy as np
import torch
import torch.utils.data as data_utils
from typing import *
from sklearn.model_selection import StratifiedKFold
import datetime
from utils.bag_utils import inference_collate, BagsLoader
from torch import optim

from sklearn.metrics import roc_auc_score


def set_seed(seed: Optional[int] = 1):

    """
    Method to set global training seed for repeatability of experiment

    :param seed: seed number
    :return: none
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def SplitBag(n_splits: Optional[int] = 5,
             num_bag: Optional[int] = 500,
             bag_labels: Optional[list] = None
             ):
    r"""
    Instance method to split dataset

        Args:
            n_splits (int): number of splits of StratifiedKFold
            num_bag (int): number of total bag datasets
            bag_labels (list): bag labels

        Return:
            train_bag_idx (list): list of index of train bag
            val_bag_idx (list): list of index of validation bag
            test_bag_idx (list): list of index of test bag
    """

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=666)
    bag_idx = range(num_bag)
    bag_label = torch.stack([torch.max(item) for item in bag_labels]).float()

    train_bag_idx = []
    val_bag_idx = []
    test_bag_idx = []

    for tmp_train, test in skf.split(bag_idx, bag_label):

        skf1 = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=666)
        tmp_label = bag_label[tmp_train]

        test_bag_idx.append(test)

        for train, val in skf1.split(tmp_train, tmp_label):
            break

        train_bag_idx.append(train)
        val_bag_idx.append(val)

    return train_bag_idx, val_bag_idx, test_bag_idx


def genSuffix(config: Optional[dict]=None):

    """
    Instance method to generate suffix
        Args:
            config (dict): config dictionary
        Return:
            suffix (str): suffix contain random seed and the experiment time
    """
    suffix = datetime.datetime.now().strftime("%y%m%d%H%M%S") + "_" + str(config['seed']) + "_" + str(config['freq']) + "_"

    return suffix


def BuildOptimizer(params, config=None):

    r"""
    instance method for building optimizer

        Args:
            params: model params
            config: optimizer config

        Return:
            none

    """
    filter_fn = filter(lambda p: p.requires_grad, params)

    if config['opt'] == 'adam':
        optimizer = optim.Adam(filter_fn, lr=config['lr'],
                               weight_decay=config['weight_decay'])
    elif config['opt'] == 'AdamW':
        optimizer = torch.optim.AdamW(filter_fn, lr=config['lr'],
                                      weight_decay=config['weight_decay'],
                                      amsgrad=config['amsgrad'])
    elif config['opt'] == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=config['lr'],
                              momentum=config['momentum'],
                              weight_decay=config['weight_decay'])
    elif config['opt'] == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=config['lr'],
                                  weight_decay=config['weight_decay'])
    elif config['opt'] == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=config['lr'],
                                  weight_decay=config['weight_decay'])
    if config['opt_scheduler'] == 'none':
        return None, optimizer
    elif config['opt_scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['opt_decay_step'],
                                              gamma=config['opt_decay_rate'])
    elif config['opt_scheduler'] == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['opt_restart'])

    return scheduler, optimizer

