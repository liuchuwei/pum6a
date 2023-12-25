#!/usr/bin/env Python
# coding=utf-8
import argparse
from argparse import ArgumentDefaultsHelpFormatter
import toml
import torch
from utils.io_utils import LoadNanoBags
import numpy as np
import pandas as pd

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

    "2.load model"
    pum6a = torch.load(config['model']['model_path'])

    "3.load dataset"
    bag = LoadNanoBags(config['dataload'])

    "4.predict"
    print("Start predict")
    Bag = bag.Bags
    bags = []
    id = Bag.keys
    for item in id:
        reads = Bag.sitedict[item]
        feature = np.stack([Bag.feature[item, :] for item in reads])
        feature = torch.tensor(feature, dtype=torch.float32)
        bags.append(feature)

    data_inst = [pum6a.Attforward(item.to(pum6a.device)) for item in bags]
    site_pro = torch.concat([item[0] for item in data_inst]).to(pum6a.device)
    site_id = id
    site_info = pd.DataFrame({'site':np.array(site_id), 'pro':site_pro.cpu().detach().numpy()})

    read_pro = torch.concat([item[1].squeeze() for item in data_inst]).to(pum6a.device)
    read_id = [Bag.id[item] for item in np.concatenate([Bag.sitedict[item] for item in id])]
    read_info = pd.DataFrame({'read':np.array(read_id), 'pro':read_pro.cpu().detach().numpy()})

    site_path = config['output'] + "_site.csv"
    site_info.to_csv(site_path, index=False)
    read_path = config['output'] + "_read.csv"
    read_info.to_csv(read_path, index=False)

    print("Finish!")