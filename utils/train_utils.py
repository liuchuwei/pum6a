import random
import numpy as np
import torch
from typing import *
from sklearn.model_selection import StratifiedKFold


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

        skf1 = StratifiedKFold(n_splits=5, shuffle=True, random_state=666)
        tmp_label = bag_label[tmp_train]

        test_bag_idx.append(test)

        for train, val in skf1.split(tmp_train, tmp_label):
            break

        train_bag_idx.append(train)
        val_bag_idx.append(train)

    return train_bag_idx, val_bag_idx, test_bag_idx

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
        self.model = model
        self.bag = bag

        self.device = (
            "cuda"
            if torch.cuda.is_available() and config['device'] == 'cuda'
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )


    def run(self):

        "1. Split dataset: 5-fold-cross-validataion"
        self.train_idx, self.val_idx, self.test_idx = SplitBag()

        "2. Train dataset"
        
        "3. Evaluate"
