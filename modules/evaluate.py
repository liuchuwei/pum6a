#!/usr/bin/env Python
# coding=utf-8
import argparse
import multiprocessing
from argparse import ArgumentDefaultsHelpFormatter

import numpy as np
import toml
import pandas as pd
from sklearn.metrics import roc_auc_score


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

    "2.load data"
    site = pd.read_csv(config['site'])

    fl = config['groundtruth']
    ground_truth = []
    for i in open(fl, "r"):

        if i.startswith("#"):
            continue

        ele = i.rstrip().split()
        ground_truth.append("|".join(ele))

    y_gt = [item.split("|") for item in ground_truth]
    for item in y_gt:
        del item[2]
    y_gt = ["|".join(item) for item in y_gt]

    y_tmp = site['site'].tolist()
    y_tmp = [item.split("|") for item in y_tmp]
    for item in y_tmp:
        del item[-1]
    y_tmp = ["|".join(item) for item in y_tmp]
    y = [item in y_gt for item in y_tmp]

    "3.evaluate"
    y_pred = site['pro'].tolist()
    bag_auc = roc_auc_score(np.array(y, dtype=np.float32), np.array(y_pred))
    print(bag_auc)