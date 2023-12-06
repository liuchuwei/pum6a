#!/usr/bin/env Python
# coding=utf-8
import argparse
from argparse import ArgumentDefaultsHelpFormatter

import toml
from utils.train_utils import set_seed, ReTrainer, puIF_Trainer, RF_Trainer
from utils.load_dataset import LoadBag
from utils.model_utils import LoadModel
from model_factory.model_factory import pum6a, puma, iAE, puIF, RF

def argparser():
    parser = argparse.ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("--config",
                        help='Path to experiment config file.',
                        required=True)

    parser.add_argument("--save_dir",
                        help='Directory to save experiment results..',
                        required=True)

    return parser


def main(args):

    "1.load config"
    config = toml.load(args.config)

    "2.set seed"
    set_seed(config['seed'])

    "3.load dataset"
    bag = LoadBag(config['dataload'])

    "4.load model_factory"
    model = LoadModel(config['model'])

    "5.train model_factory"
    if config['model_chosen'] in ['puma', 'pum6a', 'iAE']:
        trainer = ReTrainer(config=config,
                          model=model,
                          train_bag=train_bag,
                          test_bag=test_bag)
    elif config['model_chosen'] == "puIF":
        trainer = puIF_Trainer(config=config,
                          model=model,
                          train_bag=train_bag,
                          test_bag=test_bag)
    elif config['model_chosen'] == "RF":
        trainer = RF_Trainer(config=config,
                          model=model,
                          train_bag=train_bag,
                          test_bag=test_bag)

    trainer.run()
