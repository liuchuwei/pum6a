import random
import numpy as np
import torch
from typing import *

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
        bag_idx = self.bag
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=666)

        # generate mask

        id = range(labels.shape[0])
        for train1, test in skf.split(id, labels):

            skf1 = StratifiedKFold(n_splits=5, shuffle=True, random_state=666)
            y1 = labels[train1]
            x1 = [id[i] for i in train1]
            idx_test = [id[i] for i in test]

            for train, val in skf1.split(x1, y1):
                idx_train = [x1[i] for i in train]
                idx_val =  [x1[i] for i in val]
                break

        "2. Train dataset"
        "3. Evaluate"
