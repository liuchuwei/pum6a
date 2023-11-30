#!/usr/bin/env Python
# coding=utf-8
import argparse
from argparse import ArgumentDefaultsHelpFormatter

import toml

from model.model_factory import milpuAtt_construct
from utils.train_utils import set_seed, Trainer
from utils.bag_utils import Bags

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
    if config['dataset'] == "MNIST":

        train_bag = Bags(dataset="MNIST", train=True)
        test_bag = Bags(dataset="MNIST", train=False)

    elif config['dataset'] == "construct":

        train_bag = Bags(dataset="construct", train=True)
        test_bag = Bags(dataset="construct", train=False)

    else:

        raise ValueError('Dataset not support')

    "4.load model"
    model_config = config['model']

    if config['dataset'] == "MNIST":

        # model = milpuAtt_MNIST(model_config)
        pass

    elif config['dataset'] == "construct":

        model = milpuAtt_construct(model_config)


    "5.train model"
    trainer = Trainer(config=config,
                      model=model,
                      train_bag=train_bag,
                      test_bag=test_bag)

    trainer.run()