#!/usr/bin/env Python
# coding=utf-8
import argparse
from argparse import ArgumentDefaultsHelpFormatter

import pandas as pd

from trainers.AdanTrainer import adanTrainer
import toml
from utils.train_utils import set_seed
from utils.io_utils import LoadBag, LoadModel, LoadTrainer
import torch

"1.load config"
config = toml.load("pum6a.toml")

"2.set seed"
set_seed(config['seed'])

"3.load dataset"
bag = LoadBag(config['dataload'])
# if config['dataload']['dataset'] == "MNIST" and config['model']['model_chosen'] in ['RF', 'puIF', "puma", "iAE"]:
tmp = [torch.flatten(item, start_dim=1) for item in bag.bags]
bag.bags = tmp

"4.load model"
model = LoadModel(config['model'])

"5. train model"
trainer = adanTrainer(config=config['trainer'],
                      model=model,
                      bag=bag)

trainer.run_one()

"6.Positive unlabel bag"
pu_bag = torch.concat([trainer.bag.bags[item] for item in trainer.pos_unlabel_index])
pu_label = torch.concat([trainer.bag.original_label[item] for item in trainer.pos_unlabel_index])

"7.Get attention weight and instance probability"
_, pij, _, A = trainer.model.Attforward(pu_bag.to("cuda"))
res = pd.DataFrame({'label':pu_label.numpy(),
                    'pro':pij.squeeze().cpu().detach().numpy(),
                    'att':A.squeeze().cpu().detach().numpy()})
res.to_csv("../../result/MNIST/pum6a/pu_unlabel_att_pro.csv", index=False)