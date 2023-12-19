#!/usr/bin/env Python
# coding=utf-8
import random

import numpy as np
import pandas as pd
import torch

from typing import Dict, List, Tuple, Union, Optional

import torch.functional as F
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader

from collections import defaultdict

#####################################
#
# Dataset
#
#####################################
class CommonDataset(Dataset):
    def __init__(self, X, y):
        self.x_data =  X
        self.y_data = y

        self.length = len(self.y_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.length

def inference_collate(batch):
    n_reads = torch.LongTensor([len(item[1]) for item in batch])
    features = torch.cat([item[1] for item in batch])
    current_feature = torch.stack([item[0] for item in batch])
    site_ids = np.stack([item[2] for item in batch])
    read_ids = np.concatenate([item[3] for item in batch])

    return current_feature, features, n_reads, site_ids, read_ids

class caNanoDS(Dataset):
    def __init__(self, data,
                 sitedict: Dict,
                 min_reads: Optional[int] = 0,
                 args=None,
                 id=None,
                 mod=None,
                 groundtruth=None
                 ):
        """
        
        Initialization function for the class
        
        :type id: object
        :param data: current features and matching features of each reads
        :param sitedict: dict key represents modification site, dict value prepresents index of reads.
        
        """

        self.data = data
        self.id = id
        self.min_reads = min_reads
        self.sitedict = self.TailorSiteDict(sitedict)
        self.args = args
        self.groundtruth = groundtruth
        self.mod = mod

        # tmp for study the code logit
        self.__getitem__(0)

    def TailorSiteDict(self, sitedict):
        """
        Instance method for removing sites with reads less than min_reads
        """

        for key in list(sitedict.keys()):
            item = sitedict[key]
            if len(item) < self.min_reads:
                sitedict.pop(key)

        self.keys = list(sitedict.keys())

        return sitedict


    def __getitem__(self, idx:int):
        '''
        Instance method to access features from reads belonging to the idx-th site in data_info attribute
        :param item:
        :return:
        '''

        id = self.keys[idx]
        mod_dict = {"ko":0, "wt":1}
        if self.mod == "train":
            read_id = self.id[self.sitedict[id]]
            mod = [mod_dict[item.split("_")[0]] for item in read_id]
            mod = torch.tensor(mod, dtype=torch.float32)
        elif self.mod == "predict":
            site = id

        reads = self.sitedict[id]
        feature = self.data[reads,:]
        current_feature = feature[:,:20]
        if not self.mod == "Predict":
            indices = random.sample(range(0, len(read_id)), self.min_reads)
            # current_feature = current_feature[np.random.choice(len(current_feature), self.min_reads, replace=False), :]
            current_feature = current_feature[indices, :]
            mod = mod[indices]
            ratio = torch.sum(mod)/self.min_reads

        site_feature = feature[0, 20:]

        current_feature = torch.tensor(current_feature, dtype=torch.float32)
        site_feature = torch.tensor(site_feature, dtype=torch.float32)
        if self.mod == "Predict":
            ids = self.id[reads]
            return site_feature, current_feature, site, ids
        else:
            return site_feature, current_feature, mod, ratio

    def __len__(self):
        return  len(self.sitedict)

#####################################
#
# Dataloader
#
#####################################
# from torch.utils.data._utils.collate import default_collate
# def train_collate(batch):
#     return {key: batch for key, batch
#             in zip (['site', 'read', 'y'], default_collate(batch))}

#####################################
#
# Dataload
#
#####################################
class DataLoad(object):

    def __init__(self, args):

        self.args = args
        self.mod = args.mod

        print("loading data...")
        if self.mod == "train":

            self.dl = self.preprocess(args.signal, args.motif)
            # self.groundtruth = self.load_groundtruth()

        #     self.wt = self.preprocess(args.mod, args.motif)
        #     self.ko = self.preprocess(args.unmod, args.motif)
        # elif self.mod == "Predict":
        print("finish!")

        print("reshape and split data for dataset input...")
        self.reshapeData()
        print("finish!")


    def load_groundtruth(self):

        fl = self.args.groundtruth
        ground_truth = []
        for i in open(fl, "r"):

            if i.startswith("#"):
                continue

            ele = i.rstrip().split()

            motif = ele[4]

            if motif in self.args.motif:
                ground_truth.append("|".join(ele))

        return ground_truth

    def preprocess(self, path, motif=None):

        # site_info = defaultdict(dict)
        cur_info = []
        mat_info = []
        read_info = []

        for i in open(path, "r"):
            ele = i.rstrip().split()
            if ele[2] in motif:
                # if "AAACA" == motif:
                    # id
                cur, mat, ids = self.extractCurrentAlign(ele)
                cur_info.append(cur)
                mat_info.append(mat)
                read_info.append(ids)


        return {'read':read_info, 'current':cur_info, 'matching':mat_info}

    def extractCurrentAlign(self, item):
        # if "AAACA" == motif:
        # id
        id = '|'.join([item[0], item[2], item[8], item[9]])

        # current signal
        cur_mean = [float(item) for item in item[3].split("|")]
        cur_std = [float(item) for item in item[4].split("|")]
        cur_median = [float(item) for item in item[5].split("|")]
        cur_length = [int(item) for item in item[6].split("|")]
        cur = np.stack([cur_mean, cur_std, cur_median, cur_length])

        # matching
        # base, strand, cov, q_mean, q_median, q_std, mis, ins, deli = ele[8].split("|")
        eventalign = item[10].split("|")
        site = eventalign[0] + "|" + eventalign[1]
        mat = ",".join(eventalign[2:]).split(",")
        mat = np.array([float(item) for item in mat])

        return cur, mat, id + "|" + site

    def reshapeData(self):

        '''

        Instance method for reshaping and spliting ko/wt for train、 test and validate dataset input

        '''

        # if self.mod == "Train":
            # nums = min(len(self.wt['read']), len(self.ko['read']))
            # ratio = self.args.ratio # wt:ko
            # ratio = [int(item) for item in  ratio.split(":")]

            # if ratio[0]/ratio[1]>=1:
            #     wt_nums = nums
            #     ko_nums = int(np.floor(nums*ratio[1]/ratio[0]))
            # else:
            #     ko_nums = nums
            #     wt_nums = int(np.floor(nums*ratio[1]/ratio[0]))
            #
            # wt_indice = random.sample(range(0, wt_nums), wt_nums)
            # ko_indice =  random.sample(range(0, ko_nums), ko_nums)

            # wt = np.concatenate([np.stack([np.concatenate(item) for item in self.wt['current']]), np.stack(self.wt['matching'])], axis=1)
            # ko = np.concatenate([np.stack([np.concatenate(item) for item in self.ko['current']]), np.stack(self.ko['matching'])], axis=1)

            # wt = wt[wt_indice]
            # ko = ko[ko_indice]

            # X = np.concatenate([wt, ko], axis=0)
            #
            # mean_val = np.mean(X, axis=0)
            # std_val = np.std(X, axis=0)
            # X_N = (X - mean_val)/std_val
            #
            # # normalize length and quality
            # X[:,15:25] = X_N[:,15:25]
            #
            # # site infomation
            # # wt_id = np.array(["wt_" + item for item in self.wt['read']])[wt_indice]
            # # ko_id = np.array(["ko_" + item for item in self.ko['read']])[ko_indice]
            # wt_id = np.array(["wt_" + item for item in self.wt['read']])
            # ko_id = np.array(["ko_" + item for item in self.ko['read']])
            #
            # id = np.concatenate([wt_id, ko_id])
            #
            # # split data
            # self.splitData(id, X)

        # if self.mod == "Predict":
        X = np.concatenate([np.stack([np.concatenate(item) for item in self.dl['current']]), np.stack(self.dl['matching'])], axis=1)
        mean_val = np.mean(X, axis=0)
        std_val = np.std(X, axis=0)
        X_N = (X - mean_val) / std_val

        # normalize length and quality
        X[:, 15:25] = X_N[:, 15:25]

        id = np.array([item for item in self.dl['read']])
        self.DS = self.build_DateSet(X, id)

    def splitData(self, id, X):

        """
        Instance method for splitting and building train、 validate and test dataset
        """

        # split for train, val and test dataset
        indices = random.sample(range(0, len(id)), len(id))

        test_size = int(0.2 * len(id))
        valid_size = int(0.2 * len(id))
        train_size = int(0.6 * len(id))

        train_indices = indices[:train_size]
        val_indices = indices[train_size:(train_size+valid_size)]
        test_indices = indices[(train_size+valid_size):]

        train_X = X[train_indices,]
        test_X = X[test_indices,]
        val_X = X[val_indices,]

        train_id = id[train_indices]
        test_id = id[test_indices]
        val_id = id[val_indices]

        self.trainDS = self.build_DateSet(train_X, train_id)
        self.testDS = self.build_DateSet(test_X, test_id)
        self.valDS = self.build_DateSet(val_X, val_id)

    def build_DateSet(self, data, id):

        """
        Instance method for building caNano dataset for model input
        :param data:
        :param id:
        :return:
        """

        site_dict = defaultdict(dict)
        for index, item in enumerate(id):
            # if self.mod == "Train":
            #     site = "-".join(item.split("|")[-9:])
            #     site = item.split("_")[0] + "-" + site
            # else:
            site = "|".join(item.split("|")[4:7]) + "|" + item.split("|")[3]
            if not site_dict[site]:
                site_dict[site] = [index]
            else:
                site_dict[site].append(index)
        if self.mod == "predict":
            DateSet = caNanoDS(data=data, sitedict=site_dict, args=self.args, id=id, mod="predict", min_reads=int(self.args.min_reads))
        elif self.mod == "train":
            # DateSet = caNanoDS(data=data, sitedict=site_dict, args=self.args, id=id, mod="train", groundtruth=self.groundtruth)
            DateSet = caNanoDS(data=data, sitedict=site_dict, args=self.args, id=id, mod="train")

        return DateSet

    def buildDataloder(self):

        if self.mod == "Train":
            # train_dl =  DataLoader(self.trainDS, collate_fn=train_collate, batch_size=self.args.config['train']['batch_size'],
            #                        shuffle=bool(self.args.config['train']['shuffle']), num_workers=self.args.config['train']['num_workers'])
            train_dl =  DataLoader(self.trainDS, batch_size=self.args.config['train']['batch_size'],
                                   shuffle=bool(self.args.config['train']['shuffle']), num_workers=self.args.config['train']['num_workers'])
            val_dl =  DataLoader(self.valDS, batch_size=self.args.config['train']['batch_size'],
                                   shuffle=bool(self.args.config['train']['shuffle']), num_workers=self.args.config['train']['num_workers'])
            test_dl =  DataLoader(self.testDS, batch_size=self.args.config['train']['batch_size'],
                                   shuffle=bool(self.args.config['train']['shuffle']), num_workers=self.args.config['train']['num_workers'])

            return train_dl, val_dl, test_dl

        if self.mod == "Predict":
            dl = DataLoader(self.DS, batch_size=self.args.config['train']['batch_size'],
                            shuffle=False,
                                num_workers=self.args.config['train']['num_workers'], collate_fn=inference_collate)
            return dl

