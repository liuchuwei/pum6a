from typing import *
from utils.bag_utils import Bags

def LoadDataset(config: Dict):

    """
    Method to load dataset and package it into bag dataset

        Args:
            config (dict): A dictionary containing dataset configurations.

        Return:
            bag: A bag object containing bag dataset
    """

    dataset = config['dataload']['dataset']
    num_bag = config['dataload']['num_bag']
    mean_nbag_length = config['dataload']['mean_nbag_length']
    var_nbag_length = config['dataload']['var_nbag_length']
    mean_abag_length = config['dataload']['mean_abag_length']
    var_abag_length = config['dataload']['var_abag_length']
    confactor = config['dataload']['confactor']
    n_pos = config['dataload']['n_pos']
    seed = config['dataload']['seed']

    bag = Bags(
        dataset=dataset,
        num_bag=num_bag,
        mean_nbag_length=mean_nbag_length,
        var_nbag_length=var_nbag_length,
        mean_abag_length=mean_abag_length,
        var_abag_length=var_abag_length,
        confactor=confactor,
        n_pos=n_pos,
        seed=seed
    )

    return bag