import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import *


class puma(nn.Module):

    r"""
    The puma model, attention mechanism was used for transforming instance probabilities into bag probability
    """

    def __init__(self, model_config: Dict):

        r"""
        Initialization function for the class

            Args:
                    model_config (Dict): A dictionary containing model configurations.

            Returns:
                    None
        """

        super(puma, self).__init__()

        self.model_config = model_config


    def forward(self):
        pass

class Net(nn.Module):

    r"""
    The logistic model, use for instance probability inference
    """
    def __init__(self):

        r"""
        Initialization function for the class
        """

        super(Net, self).__init__()

        self.A = torch.nn.Parameter(torch.rand(1))
        self.B = torch.nn.Parameter(torch.rand(1))

    def forward(self, inputs):

        r'''
        Instance method to get instance probability.
        '''

        return torch.sigmoid(self.A * inputs + self.B)



class milpuAttention(nn.Module):

    r"""
    The milpuAttention model, attention mechanism was used for transforming instance probabilities into bag probability
    """

    def __init__(self, model_config: Dict):

        r"""
        Initialization function for the class

            Args:
                    model_config (Dict): A dictionary containing model configurations.

            Returns:
                    None
        """

        super(milpuAttention, self).__init__()

        self.model_config = model_config

        self.build_model()


    def build_model(self):

        r'''
        Instance method for building milpuAttention model according to config

        '''

        self.L = self.model_config['attention']['L']
        self.D = self.model_config['attention']['D']
        self.K = self.model_config['attention']['K']

        self.encoder = nn.Sequential(
            # 28 x 28
            nn.Conv2d(1, 4, kernel_size=5),
            # 4 x 24 x 24
            nn.ReLU(True),
            nn.Conv2d(4, 8, kernel_size=5),
            nn.ReLU(True),
            # 8 x 20 x 20 = 3200
            nn.Flatten(),
            nn.Linear(3200, 28*28),
            # 10
            nn.Softmax(),
        )

        self.decoder = nn.Sequential(
            # 10
            nn.Linear(28*28, 2000),
            # 400
            nn.ReLU(True),
            nn.Linear(2000, 4000),
            # 4000
            nn.ReLU(True),
            nn.Unflatten(1, (10, 20, 20)),
            # 10 x 20 x 20
            nn.ConvTranspose2d(10, 10, kernel_size=5),
            # 24 x 24
            nn.ConvTranspose2d(10, 1, kernel_size=5),
            # 28 x 28
            nn.Sigmoid(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )


        self.logistic = Net()


    def autoencoder_forward(self, x):

        r"""
            Instance method to get modification probability on the site level from instance features.

                Args:
                    x (torch.Tensor): A tensor representation of the instance features

                Returns:
                    enc (torch.Tensor): A tensor representation of the autoencoder latent features
                    dec (torch.Tensor): A tensor representation of the autoencoder reconstruct features
        """

        enc = self.encoder(x)
        dec = self.decoder(enc)

        return enc, dec

    def Attforward(self, x):

        r'''
        Instance method to get modification probability on the site level from instance features.

                Args:
                        x (torch.Tensor): A tensor representation of the instance features
                Returns:
                        bp (torch.Tensor): A tensor representation the bag probability
                        pij (torch.Tensor): A tensor representation the instance probability
                        A (torch.Tensor): A tensor containing attention weight of instance features

        '''


        enc, dec = self.autoencoder_forward(x)

        l1 = torch.nn.PairwiseDistance()(torch.flatten(x, start_dim=1), torch.flatten(dec, start_dim=1))
        pij = self.logistic(l1)

        A = self.attention(enc)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        bp = 1 - torch.prod(torch.pow(1-pij+1e-10, A).clip(min=0, max=1), dim=1)

        return bp, pij, A

    def _log_diverse_density(self, pi, y_bags):
        r'''
        Instance method to Compute the likelihood given bag labels y_bags and bag probabilities pi.
                Args:
                        pi (torch.Tensor): A tensor representation of the bag probabilities
                        y_bags (torch.Tensor): A tensor representation of the bag labels
                Returns:
                        likelihood (torch.Tensor): A tensor representation of the likelihood

        '''

        z = torch.where(y_bags == -1)[0]
        if z.nelement() > 0:
            zero_sum = torch.sum(torch.log(1 - pi[z] + 1e-10))
        else:
            zero_sum = torch.tensor(0).float()

        o = torch.where(y_bags == 1)[0]
        if o.nelement() > 0:
            one_sum = torch.sum(torch.log(pi[o] + 1e-10))
        else:
            one_sum = torch.tensor(0).float()
        return zero_sum + one_sum

    def bag_forward(self, input):

        r'''
        Instance method to get modification probability on the bag level from instance features.

               Args:
                       input (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
                       Tensor representation of bag, bag_labels, instance labels and number of instance
               Returns:
                       loss (torch.Tensor): A tensor representation of model loss

       '''


        bag, bag_labels, instance_labels, n_instance = input

        idx_l1 = torch.where(torch.stack(bag_labels) != 1)[0]

        if idx_l1.shape[0] > 0:
            data_inst_l1 = torch.concat([bag[index] for index in idx_l1])
            enc, dec = self.autoencoder_forward(data_inst_l1)
            loss1 = torch.nn.MSELoss()(data_inst_l1, dec)
        else:
            loss1 = torch.tensor(0, dtype=torch.float)  # reconstruct loss

        idx_l2 = torch.where(torch.stack(bag_labels) != 0)[0]
        if idx_l2.shape[0] > 0:
            data_inst_l2 = [bag[index] for index in idx_l2]
            l2 = [self.Attforward(item) for item in data_inst_l2]
            pi = torch.stack([item[0] for item in l2])
            y = torch.stack([bag_labels[index] for index in idx_l2])
            loss2 = -1*(self._log_diverse_density(pi, y)+1e-10) + 0.01*(self.logistic.A**2+self.logistic.B**2)[0]
        else:
            loss2 = 0.01*(self.logistic.A**2+self.logistic.B**2)[0]

        return loss1 + loss2



