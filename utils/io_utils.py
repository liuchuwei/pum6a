from model_factory.PUM6A import pum6a
from typing import *
from utils.bag_utils import Bags
from utils.train_utils import adanTrainer

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
        target=target,
        seed=seed
    )

    return bag


def LoadModel(config):

    """
    Method to load model_factory according to config

        Args:
            config (dict): A dictionary containing dataset configurations.

        Return:
            model: Positive and Unlabeled Multi-Instance Model
    """

    if config['model_chosen']=='pum6a':
        model = pum6a(config)

    elif config['model_chosen']=='puma':
        pass

    elif config['model_chosen']=='iAE':
        pass

    elif config['model_chosen']=='puIF':
        pass

    elif config['model_chosen']=='RF':
        pass

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

    if config['trainer_chosen']=="adanTrainer":
        trainer = adanTrainer(config=config,
                              model=model,
                              bag=bag)


    return trainer