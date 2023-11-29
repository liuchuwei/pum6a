#!/usr/bin/env Python
# coding=utf-8
import argparse
from argparse import ArgumentDefaultsHelpFormatter

import toml

from model.model_factory import milpuAttention
from utils.train_utils import set_seed, Trainer
from utils.bag_utils import Bags

import numpy as np

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
    train_bag = Bags(dataset="MNIST", train=True)

    test_bag = Bags(dataset="MNIST", train=False)
    "4.load model"
    model_config = config['model']
    model = milpuAttention(model_config)

    "5.train model"
    trainer = Trainer(config=config,
                      model=model,
                      train_bag=train_bag,
                      test_bag=test_bag)

    trainer.run()