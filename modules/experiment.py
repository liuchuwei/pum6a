#!/usr/bin/env Python
# coding=utf-8
import argparse
from argparse import ArgumentDefaultsHelpFormatter

import toml
from utils.train_utils import set_seed
from utils.io_utils import LoadBag, LoadModel, LoadTrainer

import torch
def argparser():
    parser = argparse.ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("--config",
                        help='Path to experiment config file.',
                        required=True)

    return parser


def main(args):

    "1.load config"
    config = toml.load(args.config)

    "2.set seed"
    set_seed(config['seed'])

    "3.load dataset"
    bag = LoadBag(config['dataload'])
    # if config['dataload']['dataset'] == "MNIST" and config['model']['model_chosen'] in ['RF', 'puIF', "puma", "iAE"]:
    tmp = [torch.flatten(item, start_dim=1) for item in bag.bags]
    bag.bags = tmp

    "4.load model"
    if not config['model']['model_chosen'] in ['PU-SKC', 'puMIL', 'LSDD', 'DSDD']:
        model = LoadModel(config['model'])

    else:
        model = None

    "5.train model"
    trainer = LoadTrainer(config=config['trainer'],
                          model=model,
                          bag=bag)
    trainer.run()

