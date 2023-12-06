from typing import *
from utils.bag_utils import Bags

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
    mean_abag_length = config['mean_abag_length']
    var_abag_length = config['var_abag_length']
    confactor = config['confactor']
    n_pos = config['n_pos']
    target = config['target']
    seed = config['seed']

    bag = Bags(
        dataset=dataset,
        num_bag=num_bag,
        mean_nbag_length=mean_nbag_length,
        var_nbag_length=var_nbag_length,
        mean_abag_length=mean_abag_length,
        var_abag_length=var_abag_length,
        confactor=confactor,
        n_pos=n_pos,
        target=target,
        seed=seed
    )

    return bag