#!/usr/bin/env Python
# coding=utf-8
import argparse
from argparse import ArgumentDefaultsHelpFormatter
import numpy as np

def argparser():

    parser = argparse.ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )

    parser.add_argument("--dataset",
                        help='Dataset for preprocess. possible dataset: MNIST',
                        default="MNIST")

    parser.add_argument("--save_dir",
                        help='Directory to output training results.',
                        required=True)

    return parser


def main(args):

    "1. load dataset"
    if args.dataset not in ['MNIST']:
        raise ValueError("Invalid dataset. possible dataset: MNIST")

    if args.dataset == "MNIST":

        data = np.load("dataset/ADBench/24_mnist.npz", allow_pickle=True)
        X, y = data['X'], data['y']

    "2. split dataset"

    "3. construct bag dataset"

    "4. save"