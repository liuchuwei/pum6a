import random
import numpy as np
import torch
from typing import *
import torch.utils.data as data_utils
from torch import optim

from utils.bag_utils import inference_collate

def set_seed(seed: Optional[int] = 1):

    """
    Method to set global training seed for repeatability of experiment

    :param seed: seed number
    :return: none
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



class RF_Trainer(object):

    """
    An object class for puIF model_factory training
    """

    def __init__(self,
                 config: Optional[Dict] = None,
                 model=None, train_bag=None,
                 test_bag=None):

        r"""
        Initialization function for the class

            Args:
                    config (Dict): A dictionary containing training configurations.
                    model: Model to train
                    train_bag: Bag dataset for model_factory training
                    test_bag: Bag dataset for model_factory testing

            Returns:
                    None
        """

        self.config = config
        self.model = model
        self.train_bag = train_bag
        self.test_bag = test_bag

    def run(self):
        '1.Build random forest model_factory'
        X = torch.concat(self.train_bag.bags)
        label_len = [len(item) for item in self.train_bag.bags]
        Y = []
        for idx, item in enumerate(self.train_bag.bags_labels):
            Y.append(item.repeat(label_len[idx]))
        Y = torch.concat(Y)

        from sklearn.ensemble import RandomForestRegressor
        clf = RandomForestRegressor(max_depth=2, random_state=888)
        clf.fit(X, Y)

        "2.Testing result"
        ins_pro = []
        bag_pro = []
        for item in self.test_bag.bags:

            p = clf.predict(item)
            p = torch.Tensor(p)
            ins_pro.append(p)
            bag_pro.append(self.model._weightnoisyor(p))

        from sklearn.metrics import roc_auc_score
        inst_y = torch.concat(self.test_bag.labels).float()
        inst_pro = torch.concat(ins_pro).squeeze().detach().numpy()

        bag_y = self.test_bag.bags_labels
        bag_auc = roc_auc_score(bag_y.cpu(), torch.concat(bag_pro).cpu().squeeze().detach().numpy())
        ins_auc = roc_auc_score(inst_y.cpu(), inst_pro)

        print(f"Bag_auc: {(100 * bag_auc):>0.1f}%, "
              f"Instance_auc: {(100 * ins_auc):>0.1f}%.")

class puIF_Trainer(object):

    """
    An object class for puIF model_factory training
    """

    def __init__(self,
                 config: Optional[Dict] = None,
                 model=None, train_bag=None,
                 test_bag=None):

        r"""
        Initialization function for the class

            Args:
                    config (Dict): A dictionary containing training configurations.
                    model: Model to train
                    train_bag: Bag dataset for model_factory training
                    test_bag: Bag dataset for model_factory testing

            Returns:
                    None
        """

        self.config = config
        self.model = model
        self.train_bag = train_bag
        self.test_bag = test_bag
        self.scheduler, self.optimizer = self.build_optimizer(params=model.parameters())

    def build_optimizer(self, params, weight_decay=0.0):

        r"""
        instance method for building optimizer

            Args:
                params: model_factory params
                weight_decay: learning rate weight decay

            Return:
                none

        """
        filter_fn = filter(lambda p: p.requires_grad, params)
        if self.config['optimizer']['opt'] == 'adam':
            optimizer = optim.Adam(filter_fn, lr=self.config['optimizer']['lr'],
                                   weight_decay=self.config['optimizer']['weight_decay'])
        elif self.config['optimizer']['opt'] == 'AdamW':
            optimizer = torch.optim.AdamW(filter_fn, lr=self.config['optimizer']['lr'],
                                          weight_decay=self.config['optimizer']['weight_decay'],
                                          amsgrad=self.config['optimizer']['amsgrad'])
        elif self.config['optimizer']['opt'] == 'sgd':
            optimizer = optim.SGD(filter_fn, lr=self.config['optimizer']['lr'],
                                  momentum=self.config['momentum']['weight_decay'],
                                  weight_decay=self.config['optimizer']['weight_decay'])
        elif self.config['optimizer']['opt'] == 'rmsprop':
            optimizer = optim.RMSprop(filter_fn, lr=self.config['optimizer']['lr'],
                                      weight_decay=self.config['optimizer']['weight_decay'])
        elif self.config['optimizer']['opt'] == 'adagrad':
            optimizer = optim.Adagrad(filter_fn, lr=self.config['optimizer']['lr'],
                                      weight_decay=self.config['optimizer']['weight_decay'])
        if self.config['optimizer']['opt_scheduler'] == 'none':
            return None, optimizer
        elif self.config['optimizer']['opt_scheduler'] == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.config['optimizer']['opt_decay_step'],
                                                  gamma=self.config['optimizer']['opt_decay_rate'])
        elif self.config['optimizer']['opt_scheduler'] == 'cos':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config['optimizer']['opt_restart'])

        return scheduler, optimizer

    def run(self):

        "1.Train isolation model_factory"
        X = torch.concat(self.train_bag.bags)
        self.model.clf.fit(X)

        "2.Get instance class"
        y = torch.Tensor(self.model.clf.predict(X))

        "3.Train nontraditional classifier"
        for t in range(self.config['epochs']):
            self.model.train()
            print(f"Epoch {t + 1}\n-------------------------------")
            lin = self.model.linear(X)
            ps = 1/(1+self.model.B**2 + torch.exp(-lin))
            y[torch.where(y == -1)] = 0
            loss = torch.sum(-1. * (y * torch.log(ps) + (1. - y) * torch.log(1. - ps)))
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            print(f"likihood_loss: {(loss):>0.1f}, ")
        print("Done!")

        "4.Testing result"
        self.model.eval()
        ins_pro = []
        bag_pro = []
        for item in self.test_bag.bags:
            lin = self.model.linear(item)
            ps = 1/(1+self.model.B**2 + torch.exp(-lin))
            c_hat = 1/(1+self.model.B**2)
            p = ps/c_hat
            ins_pro.append(p)
            bag_pro.append(self.model._weightnoisyor(p))

        from sklearn.metrics import roc_auc_score
        inst_y = torch.concat(self.test_bag.labels).float()
        inst_pro = torch.concat(ins_pro).squeeze().detach().numpy()

        bag_y = self.test_bag.bags_labels
        bag_auc = roc_auc_score(bag_y.cpu(), torch.concat(bag_pro).cpu().squeeze().detach().numpy())
        ins_auc = roc_auc_score(inst_y.cpu(), inst_pro)

        print(f"Bag_auc: {(100 * bag_auc):>0.1f}%, "
              f"Instance_auc: {(100 * ins_auc):>0.1f}%.")

class ReTrainer(object):

    """
    An object class for model_factory training with self-adaptive process to select most reliable negative bags
    """

    def __init__(self,
                 config: Optional[Dict] = None,
                 model=None, train_bag=None,
                 test_bag=None):

        r"""
        Initialization function for the class

            Args:
                    config (Dict): A dictionary containing training configurations.
                    model: Model to train
                    train_bag: Bag dataset for model_factory training
                    test_bag: Bag dataset for model_factory testing

            Returns:
                    None
        """

        self.config = config
        self.model = model
        self.train_bag = train_bag
        self.test_bag = test_bag

        self.device = (
            "cuda"
            if torch.cuda.is_available() and config['device'] == 'cuda'
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        self.generateDataLoader()
        self.initNegLabel()
        self.scheduler, self.optimizer = self.build_optimizer(params=model.parameters())
        self.model.to(self.device)
        self.obtain_cont_factor()

    def obtain_cont_factor(self):
        r"""
        instance method to obtain_cont_factor
        """

        if not isinstance(self.config['confactor'], str):
            self.model.cont_factor = self.config['confactor']
        else:
            tot_inst = len(torch.concat(self.train_bag.labels))
            inst_labeledbags = np.sum(
                [len(self.train_bag.labels[item]) for item in (np.where(self.train_bag.bags_labels == 1)[0])])
            inst_unlabeledbags = np.sum(
                [len(self.train_bag.labels[item]) for item in (np.where(self.train_bag.bags_labels != 1)[0])])
            self.model.cont_factor = max((0.3 * tot_inst - 0.25 * inst_labeledbags) / inst_unlabeledbags, 0)

    def generateDataLoader(self):
        """
        Instance method for generating dataloader

        """
        self.train_loader = data_utils.DataLoader(self.train_bag,
                                                  batch_size=self.config['batch_size'],
                                                  shuffle=True,
                                                  collate_fn=inference_collate)

        self.test_loader = data_utils.DataLoader(self.test_bag,
                                                 batch_size=self.config['batch_size'],
                                                 shuffle=True,
                                                 collate_fn=inference_collate)

    def build_optimizer(self, params, weight_decay=0.0):

        r"""
        instance method for building optimizer

            Args:
                params: model_factory params
                weight_decay: learning rate weight decay

            Return:
                none

        """
        filter_fn = filter(lambda p: p.requires_grad, params)
        if self.config['optimizer']['opt'] == 'adam':
            optimizer = optim.Adam(filter_fn, lr=self.config['optimizer']['lr'],
                                   weight_decay=self.config['optimizer']['weight_decay'])
        elif self.config['optimizer']['opt'] == 'AdamW':
            optimizer = torch.optim.AdamW(filter_fn, lr=self.config['optimizer']['lr'],
                                          weight_decay=self.config['optimizer']['weight_decay'],
                                          amsgrad=self.config['optimizer']['amsgrad'])
        elif self.config['optimizer']['opt'] == 'sgd':
            optimizer = optim.SGD(filter_fn, lr=self.config['optimizer']['lr'],
                                  momentum=self.config['momentum']['weight_decay'],
                                  weight_decay=self.config['optimizer']['weight_decay'])
        elif self.config['optimizer']['opt'] == 'rmsprop':
            optimizer = optim.RMSprop(filter_fn, lr=self.config['optimizer']['lr'],
                                      weight_decay=self.config['optimizer']['weight_decay'])
        elif self.config['optimizer']['opt'] == 'adagrad':
            optimizer = optim.Adagrad(filter_fn, lr=self.config['optimizer']['lr'],
                                      weight_decay=self.config['optimizer']['weight_decay'])
        if self.config['optimizer']['opt_scheduler'] == 'none':
            return None, optimizer
        elif self.config['optimizer']['opt_scheduler'] == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.config['optimizer']['opt_decay_step'],
                                                  gamma=self.config['optimizer']['opt_decay_rate'])
        elif self.config['optimizer']['opt_scheduler'] == 'cos':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config['optimizer']['opt_restart'])

        return scheduler, optimizer

    def initNegLabel(self):

        """
        Instance method for initiating negative bag label
        """

        y_tmp = torch.clone(self.train_bag.bags_labels)
        neg_idx = torch.where(y_tmp == 0)[0]
        n_neg = self.train_bag.n_pos
        y_tmp[neg_idx[torch.randperm(neg_idx.size(0))[:n_neg]]] = -1

        self.y_tmp = y_tmp.to(self.device)

    def refreshNegLabel(self, bag_scores):

        """
        Instance method for refreshing negative bag label

            Args:
                bag_scores: bag probability obtaining during training

            Returns:
                none
        """

        nonpos_idx = torch.where(self.train_bag.bags_labels == 0)[0].to(self.device)
        sorted_idx = torch.argsort(bag_scores[nonpos_idx], dim=0)[:self.train_bag.n_pos]
        self.y_tmp = torch.clone(self.train_bag.bags_labels).to(self.device)
        self.y_tmp[nonpos_idx[sorted_idx]] = -1

    def train_epoch(self):
        r"""
        Instance method for taining one epoch
        """

        size = len(self.train_loader.dataset)

        self.model.train()

        bag_scores = torch.zeros([self.train_bag.num_bag, 1]).to(self.device).float()

        for batch, (features, n_instance, bag_idx) in enumerate(self.train_loader):

            idx = [np.sum(n_instance[:it]) for it in range(1, len(n_instance) + 1)]
            bag = np.split(features, idx)[:-1]
            bag = [item.to(self.device) for item in bag]

            bag_labels = self.y_tmp[bag_idx]
            bag_labels = bag_labels.to(self.device)

            loss2, data_inst = self.model.bag_forward((bag, bag_labels, n_instance))
            data_inst = data_inst[:len(bag)]
            loss = loss2
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # bag_scores[bag_idx] = torch.concat([item[0] for item in data_inst])
            # if self.config['model_chosen'] == 'puma':
            #     bag_scores[bag_idx] = torch.stack([item[0] for item in data_inst])
            # elif self.config['model_chosen'] == 'pum6a':
            bag_scores[bag_idx] = torch.stack([item[0] for item in data_inst])
            # bag_scores[bag_idx] = torch.stack([torch.max(item[1]) for item in data_inst]).unsqueeze(1)

            if batch % 5 == 0:
                loss, current = loss.item(), (batch + 1) * self.config['batch_size']
                # print(f"rec_loss: {loss1:>7f} likehood_loss_p: {loss2:>7f}  "
                #       f"likehood_loss_c: {loss3:>7f} [{current:>5d}/{size:>5d}]")
                #
                print(f"likehood_loss_p: {loss2:>7f}  "
                      f"[{current:>5d}/{size:>5d}]")

        self.refreshNegLabel(bag_scores)

        # self.model_factory.eval()
        # data_inst = [self.model_factory.Attforward(item.to(self.device)) for item in self.train_bag.pos_bag]
        # att = [item[1] for item in data_inst]
        # neg_bag = []
        #
        # for idx, item in enumerate(att):
        #
        #     if 0 < torch.sum(item >= 0.3) < len(item[0]):
        #         neg_bag.append(self.train_bag.pos_bag[idx].to(self.device)[(item < 0.2).squeeze()])
        #
        # self.model_factory.pesu_bag = neg_bag

    def _process_decision_scores(self, threshold_: Optional[float] = 0.5, data_inst=None, bag_idx=None):

        self.threshold_ = threshold_

        bag_pro_c = torch.stack([item[0] for item in data_inst])
        bag_pro_c = (bag_pro_c > self.threshold_).float().squeeze()
        # bag_pro_p = data_inst[1]
        bag_pro_p = torch.stack([item[1] for item in data_inst])
        bag_pro_p = (bag_pro_p > self.threshold_).float().squeeze()
        # inst_pro = torch.concat([torch.flatten(item) for item in data_inst[2]])
        inst_pro = torch.concat([item[2] for item in data_inst])
        inst_pro = (inst_pro > self.threshold_).float()

        inst_y = [self.test_bag.labels[item] for item in bag_idx]
        bag_y = torch.stack([torch.max(item).float() for item in inst_y]).to(self.device)
        inst_y = torch.concat(inst_y).float().to(self.device)

        bag_correct_c = (bag_pro_c == bag_y).sum().type(torch.float)
        # bag_correct_c = 0
        bag_correct_p = (bag_pro_p == bag_y).sum().type(torch.float)
        inst_correct = (inst_pro == inst_y).sum().type(torch.float)
        inst_len = len(inst_y)

        return bag_correct_c, bag_correct_p, inst_correct, inst_len

    def test_epoch(self):

        r"""
        Instance method for testing one epoch
        """

        self.model.eval()
        size = len(self.test_loader.dataset)
        num_batches = len(self.test_loader)
        test_loss, bag_correct_c, bag_correct_p, inst_correct = 0, 0, 0, 0
        inst_len = 0
        bag_pro_att = []
        bag_pro = []
        bag_y = []
        inst_pro = []
        att_pro = []
        inst_y = []
        original_y = []
        with torch.no_grad():
            for features, n_instance, bag_idx in self.test_loader:
                idx = [np.sum(n_instance[:it]) for it in range(1, len(n_instance) + 1)]
                bag = np.split(features, idx)[:-1]
                bag = [item.to(self.device) for item in bag]

                bag_labels = self.test_bag.bags_labels[bag_idx]
                bag_labels = bag_labels.to(self.device)

                loss2, data_inst = self.model.bag_forward((bag, bag_labels, n_instance))
                data_inst = data_inst[:len(bag)]
                test_loss += loss2

                # bag_cor_c, bag_cor_p, inst_cor, inst_num = self._process_decision_scores(
                #     data_inst=data_inst, bag_idx=bag_idx)

                bag_pro.append(torch.concat([item[0] for item in data_inst]))

                if self.config['model_chosen'] == 'pum6a':
                    bag_pro_att.append(torch.concat([item[2] for item in data_inst]))
                bag_y.append(self.test_bag.bags_labels[bag_idx])

                inst_pro.append(torch.concat([item[1].squeeze() for item in data_inst]))
                if self.config['model_chosen'] == 'pum6a':
                    att_pro.append(torch.concat([item[3].squeeze() for item in data_inst]))

                inst_y.append(torch.concat([self.test_bag.labels[item] for item in bag_idx]).float())
                original_y.append(torch.concat([self.test_bag.original_label[item] for item in bag_idx]).float())


        from sklearn.metrics import roc_auc_score

        if self.config['model_chosen'] == 'pum6a':
            bag_auc_att = roc_auc_score(torch.concat(bag_y).cpu(), torch.concat(bag_pro_att).cpu())
            att_auc = roc_auc_score(torch.concat(inst_y).cpu(), torch.concat(att_pro).cpu())

        bag_auc = roc_auc_score(torch.concat(bag_y).cpu(), torch.concat(bag_pro).cpu())
        ins_auc = roc_auc_score(torch.concat(inst_y).cpu(), torch.concat(inst_pro).cpu())
        # bag_pi = self.decision_function()
        # bag_pi = bag_pi.reshape(-1)
        # bag_auc = roc_auc_score(torch.concat(bag_y).cpu(), bag_pi)
        import pandas as pd
        if self.config['model_chosen'] == 'pum6a':
            df = pd.DataFrame({'Att': torch.concat(inst_pro).cpu(),
                               'type': torch.concat(original_y).cpu()})
            group1 = df.groupby('type')

        test_loss /= num_batches
        if self.config['model_chosen'] == 'pum6a':

            print(f"Bag_auc: {(100 * bag_auc):>0.1f}%, "
                  f"Bag_auc_att: {(100 * bag_auc_att):>0.1f}%, "
                  f"Instance_auc: {(100 * ins_auc):>0.1f}%, "
                  f"Instance_auc_att: {(100 * att_auc):>0.1f}%, "
                  f"Avg loss: {test_loss:>8f} \n")
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 5000)
            print(group1.mean().T)

        else:
            print(f"Bag_auc: {(100 * bag_auc):>0.1f}%, "
                  f"Instance_auc: {(100 * ins_auc):>0.1f}%, "
                  f"Avg loss: {test_loss:>8f} \n")
    def decision_function(self):

        # enable the evaluation mode
        self.model.eval()
        X_bags = torch.stack(self.test_bag.bags)
        # construct the vector for holding the reconstruction error
        b_scores = torch.zeros([X_bags.shape[0], 1]).to(self.device, non_blocking=True).float()
        i_scores = torch.zeros([X_bags.shape[0] * X_bags.shape[1], 1]).to(self.device, non_blocking=True).float()

        with torch.no_grad():
            self.n_samples = 10
            self.n_features = 2

            for features, n_instance, bag_idx in self.test_loader:
                idx = [np.sum(n_instance[:it]) for it in range(1, len(n_instance) + 1)]
                data = np.split(features, idx)[:-1]
                data = torch.stack(data).to(self.device)
                # data = [item.to(self.device) for item in data]
                local_batch_size, _, _ = data.size()
                data_idx = torch.tensor(bag_idx).to(self.device, non_blocking=True)
                # mi = data_idx[0]
                # ma = data_idx[local_batch_size - 1] + 1
                data_inst = torch.reshape(data, (local_batch_size * self.n_samples, self.n_features))
                data_inst = data_inst.to(self.device, non_blocking=True).float()
                l1 = torch.nn.PairwiseDistance(p=2, eps=0)(data_inst, self.model.autoencoder_forward(data_inst)[1])
                l1 = torch.reshape(l1, (local_batch_size, self.n_samples, 1))
                instance_scr = self.model._logistic(l1)
                # i_scores[mi * self.n_samples:ma * self.n_samples] = instance_scr.reshape(
                #     local_batch_size * self.n_samples, 1)
                pi = self.model._weightnoisyor(instance_scr)
                b_scores[data_idx] = pi.reshape(local_batch_size, 1)

        # return b_scores.cpu().numpy(), i_scores.cpu().numpy()
        return b_scores.cpu().numpy()

    def run(self):

        for t in range(self.config['epochs']):
            print(f"Epoch {t + 1}\n-------------------------------")
            self.train_epoch()
            self.test_epoch()

        print("Done!")

