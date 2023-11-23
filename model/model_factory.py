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

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(self.model_config['extractor_1']['Conv2d_1st'][0],
                      self.model_config['extractor_1']['Conv2d_1st'][1],
                      kernel_size=self.model_config['extractor_1']['Conv2d_1st'][2]),
            nn.ReLU(),
            nn.MaxPool2d(self.model_config['extractor_1']['MaxPool2d_1st'][0],
                         stride=self.model_config['extractor_1']['MaxPool2d_1st'][1]),
            nn.Conv2d(self.model_config['extractor_1']['Conv2d_2nd'][0],
                      self.model_config['extractor_1']['Conv2d_2nd'][1],
                      kernel_size=self.model_config['extractor_1']['Conv2d_2nd'][2]),
            nn.ReLU(),
            nn.MaxPool2d(self.model_config['extractor_1']['MaxPool2d_2nd'][0],
                         stride=self.model_config['extractor_1']['MaxPool2d_2nd'][1]), )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(self.model_config['extractor_2']['Layer'][0], self.model_config['extractor_2']['Layer'][1]),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        r'''
        Instance method to get modification probability on the site level from instance features.

                Args:
                        x (torch.Tensor): A tensor representation of the instance features
                Returns:
                        Y_prob (torch.Tensor): A tensor containing modification probability on the bag level
                        Y_hat (torch.Tensor): Binary format of Y_prob
                        A (torch.Tensor): A tensor containing attention weight of instance features
        '''
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, 50 * 4 * 4)
        H = self.feature_extractor_part2(H)  # NxL

        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A
