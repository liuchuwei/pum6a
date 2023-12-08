from model_factory.PUM6A import pum6a
from model_factory.PUMA import puma
from model_factory.IAE import iAE
from model_factory.PUIF import puIF
from model_factory.RandomForest import RF
from typing import *

from trainers.RandomFroestTrainer import RF_Trainer
from utils.bag_utils import Bags
from trainers.AdanTrainer import adanTrainer
from trainers.PUIF_Trainer import puIF_Trainer

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

    if config['model_chosen'] == 'pum6a':
        model = pum6a(config)

    elif config['model_chosen'] == 'puma':
        model = puma(config)

    elif config['model_chosen'] == 'iAE':
        model = iAE(config)

    elif config['model_chosen'] == 'puIF':
        model = puIF(config)

    elif config['model_chosen'] == 'RF':
        model = RF(config)

    elif config['model_chosen'] == 'PUSKC':
        pass

    elif config['model_chosen'] == 'PUMIL':
        pass

    elif config['model_chosen'] == 'LSDD':
        pass

    elif config['model_chosen'] == 'DSDD':
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
        return trainer