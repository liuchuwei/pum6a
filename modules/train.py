#!/usr/bin/env Python
# coding=utf-8
import argparse
from argparse import ArgumentDefaultsHelpFormatter
import toml

from utils.train_utils import set_seed
from utils.io_utils import LoadNanoBags, LoadModel, LoadTrainer

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
    bag = LoadNanoBags(config['dataload'])

    "4.load model"
    model = LoadModel(config['model'])

    "5.train model"
    trainer = LoadTrainer(config=config['trainer'],
                          model=model,
                          bag=bag)

    trainer.run()