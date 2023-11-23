import torch.utils.data as data_utils
from utils.bag_utils import Bags
import torch.nn as nn
from typing import *
import toml


# load data
train_loader = data_utils.DataLoader(Bags(dataset="Cifar10", train=True),
                                     batch_size=1,
                                     shuffle=True)

test_loader = data_utils.DataLoader(Bags(dataset="Cifar10", train=False),
                                     batch_size=1,
                                     shuffle=True)

# build model
class puAttention(nn.Module):

    r"""
    The puAttention model, attention mechanism was used for transforming instance probabilities into bag probability
    """

    def __init__(self, model_config: Dict):

        r"""
        Initialization function for the class

            Args:
                    model_config (Dict): A dictionary containing model configurations.

            Returns:
                    None
        """

        super(puAttention, self).__init__()

        self.model_config = model_config


    def forward(self):
        pass

model_config = toml.load('config/MNIST_puAttention.toml')
model = puAttention(model_config=model_config)

# train & test

# tmp
import os
import numpy