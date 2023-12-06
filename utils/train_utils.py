import os
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


def genSuffix(seed: Optional[int]=88888888):

    """
    Instance method to generate suffix
        Args:
            seed (int): random seed of of the experiment
        Return:
            suffix (str): suffix contain random seed and the experiment time
    """
    suffix = datetime.datetime.now().strftime("%y%m%d%H%M%S") + "_" + str(seed) + "_"

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

class adanTrainer(object):

    """
    An object class for model training with self-adaptive process to select most reliable negative bags
    """

    def __init__(self,
                 config: Optional[Dict] = None,
                 model=None, bag=None):

        r"""
        Initialization function for the class

            Args:
                    config (Dict): A dictionary containing training configurations.
                    model: Model to train
                    bag: Bag dataset input

            Returns:
                    None
        """

        self.config = config
        self.init_model = model
        self.bag = bag
        self.n_splits = config['n_splits']
        self.suffix = genSuffix(config['seed'])

        self.device = (
            "cuda"
            if torch.cuda.is_available() and config['device'] == 'cuda'
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )


    def saveInit(self):

        """
        Instance method to create output path and save initiate model
        """

        tmp_dir = self.config['save_dir'].split('/')
        cur_dir = os.getcwd()

        for item in tmp_dir:

            cur_dir = cur_dir + "/" + item

            if not os.path.exists(cur_dir):
                os.mkdir(cur_dir)

        self.init_path = self.config['save_dir'] + "/" + self.suffix+ "init_model.pt"
        torch.save(self.init_model, self.init_path)

    def initNegLabel(self):

        """
        Instance method for initiating negative bag label
        """

        y_tmp = torch.clone(self.train_bag_label)
        neg_idx = torch.where(y_tmp == 0)[0]
        n_neg = torch.sum(y_tmp).int()
        y_tmp[neg_idx[torch.randperm(neg_idx.size(0))[:n_neg]]] = -1

        self.y_tmp = y_tmp.to(self.device)

    def refreshNegLabel(self, bag_scores):

        """
        Instance method for refreshing negative bag label

            Args:
                bag_scores: bag probability obtaining during training

            Returns:
                none
        """

        nonpos_idx = torch.where(self.train_bag_label == 0)[0].to(self.device)
        sorted_idx = torch.argsort(bag_scores[nonpos_idx], dim=0)[:self.config['n_pos']]
        self.y_tmp = torch.clone(self.train_bag_label).to(self.device)
        self.y_tmp[nonpos_idx[sorted_idx]] = -1

    def train_epoch(self):
        r"""
        Instance method for taining one epoch
        """

        size=len(self.train_bag)
        self.model.train()
        bag_scores = torch.zeros([len(self.train_bag), 1]).to(self.device).float()

        for batch, (features, n_instance, bag_idx) in enumerate(self.train_loader):

            idx = [np.sum(n_instance[:it]) for it in range(1, len(n_instance) + 1)]
            bag = np.split(features, idx)[:-1]
            bag = [item.to(self.device) for item in bag]

            bag_labels = self.y_tmp[bag_idx]
            bag_labels = bag_labels.to(self.device)

            loss, data_inst = self.model.bag_forward((bag, bag_labels, n_instance))
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            bag_scores[bag_idx] = torch.stack([item[0] for item in data_inst])

            if batch % 5 == 0:
                loss, current = loss.item(), (batch + 1) * self.config['batch_size']
                print(f"likehood_loss_p: {loss:>7f}  "
                      f"[{current:>5d}/{size:>5d}]")

        self.refreshNegLabel(bag_scores)

    def val_epoch(self):
        r"""
        Instance method for validate one epoch

        Return:
            bag_loss: likelihood loss of validation dataset
        """
        self.model.eval()
        bag_loss = self.model.validation(self.val_bag, self.val_bag_label.to(self.device))

        return bag_loss

    def tesing(self):
        r"""
        Instance method for testing

        Return:
            bag_auc: auc score of bag
            ins_auc: auc score of instance
        """

        self.model.eval()
        bag_pro, ins_pro = self.model.decision(self.test_bag)
        bag_y = torch.stack([item.max() for item in self.test_bag_label]).float()
        ins_y = torch.concat([item.squeeze() for item in self.test_bag_label]).float()

        bag_auc = roc_auc_score(bag_y, bag_y.cpu().detach().numpy())
        ins_auc = roc_auc_score(ins_y, ins_pro.cpu().detach().numpy())

        return bag_auc, ins_auc

    def run(self):

        "1. Split dataset: 5-fold-cross-validation"
        self.train_idx, self.val_idx, self.test_idx = SplitBag(n_splits=self.n_splits,
                                                               num_bag=self.bag.num_bag,
                                                               bag_labels=self.bag.labels)

        "2. Save initial model"
        self.saveInit()

        "3. train and save model"
        for idx in range(self.n_splits):

            self.model = torch.load(self.init_path)
            self.model.eval()

            self.train_bag = [self.bag.bags[item] for item in self.train_idx[idx]]
            self.val_bag = [self.bag.bags[item] for item in self.val_idx[idx]]
            self.train_loader = data_utils.DataLoader(BagsLoader(self.train_bag),
                                                  batch_size=self.config['batch_size'],
                                                  shuffle=True,
                                                  collate_fn=inference_collate)

            train_bag_label = torch.stack([self.bag.labels[item].max() for item in self.train_idx[idx]]).float()
            val_bag_label = torch.stack([self.bag.labels[item].max() for item in self.val_idx[idx]]).float()
            total_label = torch.concat([train_bag_label, val_bag_label])
            n_pos = self.config['n_pos']
            pos_idx = np.random.choice(torch.where(total_label == 1)[0], size=n_pos, replace=False)
            s = torch.zeros(len(total_label))
            s[pos_idx] = 1

            self.train_bag_label = s[:len(train_bag_label)]
            self.val_bag_label = s[len(train_bag_label):]

            self.initNegLabel()


            self.test_bag = [self.bag.bags[item] for item in self.test_idx[idx]]
            self.test_bag_label = [self.bag.labels[item] for item in self.test_idx[idx]]

            self.scheduler, self.optimizer = BuildOptimizer(params=self.model.parameters(),
                                            config=self.config['optimizer'])

            best = 88888888
            for t in range(self.config['epochs']):

                print(f"Epoch {t + 1}\n-------------------------------")
                self.train_epoch()
                cost = self.val_epoch()
                print(f"val_bag_loss: {(cost):>0.1f}")

                if cost < best:
                    best = cost
                    patience = 0
                else:
                    patience += 1

                # if patience == self.config['early_stopping']:
                #     break
                bag_auc, ins_auc = self.tesing()

                print(f"bag_auc: {(100 * bag_auc):>0.1f}%, "
                      f"ins_auc: {(100 * ins_auc):>0.1f}%")

            print("Done!")

