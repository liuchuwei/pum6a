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

        # logistic model
        self.model.A = torch.nn.Parameter(torch.tensor(torch.rand(1)), requires_grad=True)
        self.model.A.grad = torch.tensor(torch.rand(1))
        self.model.B = torch.nn.Parameter(torch.tensor(torch.rand(1)), requires_grad=True)

    def _logistic(self, loss):

        r'''
            Instance method to get instance probability.
        '''

        return torch.sigmoid(self.model.A * loss + self.model.B)

    def forward(self, x):

        r'''
        Instance method to get modification probability on the site level from instance features.

                Args:
                        x (torch.Tensor): A tensor representation of the instance features
                Returns:
                        pi (torch.Tensor): A tensor representation the instance probability
                        dec (torch.Tensor): A tensor representation of the restruct feature

        '''

        x = x.squeeze(0)

        enc = self.encoder(x)
        dec = self.decoder(enc)

        x_fla = torch.flatten(enc, start_dim=1)
        l1 = torch.nn.PairwiseDistance()(x_fla, enc)
        pij = self._logistic(l1)

        A = self.attention(enc)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        bag_prob = 1 - torch.prod(torch.pow(1-pij+1e-10, A).clip(min=0, max=1), dim=1)

        return bag_prob, pij, A, dec

    def bag_forward(self, input):

        r'''
           Instance method to get modification probability on the bag level from instance features.

                   Args:
                           input (torch.Tensor, torch.Tensor): Tensor representation of the bag features and bag labels
                   Returns:
                           Y_prob (torch.Tensor): A tensor containing modification probability on the bag level
                           Y_hat (torch.Tensor): Binary format of Y_prob
                           A (torch.Tensor): A tensor containing attention weight of instance features
           '''


        bag, bag_labels = input

        bag_pro = [self.forward(item) for item in bag]

#
#
# class milpuAttention(nn.Module):
#
#     r"""
#     The milpuAttention model, attention mechanism was used for transforming instance probabilities into bag probability
#     """
#
#     def __init__(self, model_config: Dict):
#
#         r"""
#         Initialization function for the class
#
#             Args:
#                     model_config (Dict): A dictionary containing model configurations.
#
#             Returns:
#                     None
#         """
#
#         super(milpuAttention, self).__init__()
#
#         self.model_config = model_config
#         self.build_model()
#
#     def build_model(self):
#
#         r'''
#         Instance method for building milpuAttention model according to config
#
#         '''
#
#         self.L = self.model_config['attention']['L']
#         self.D = self.model_config['attention']['D']
#         self.K = self.model_config['attention']['K']
#
#         self.feature_extractor_part1 = nn.Sequential(
#             nn.Conv2d(self.model_config['extractor_1']['Conv2d_1st'][0],
#                       self.model_config['extractor_1']['Conv2d_1st'][1],
#                       kernel_size=self.model_config['extractor_1']['Conv2d_1st'][2]),
#             nn.ReLU(),
#             nn.MaxPool2d(self.model_config['extractor_1']['MaxPool2d_1st'][0],
#                          stride=self.model_config['extractor_1']['MaxPool2d_1st'][1]),
#             nn.Conv2d(self.model_config['extractor_1']['Conv2d_2nd'][0],
#                       self.model_config['extractor_1']['Conv2d_2nd'][1],
#                       kernel_size=self.model_config['extractor_1']['Conv2d_2nd'][2]),
#             nn.ReLU(),
#             nn.MaxPool2d(self.model_config['extractor_1']['MaxPool2d_2nd'][0],
#                          stride=self.model_config['extractor_1']['MaxPool2d_2nd'][1]), )
#
#         self.feature_extractor_part2 = nn.Sequential(
#             nn.Linear(self.model_config['extractor_2']['Layer'][0], self.model_config['extractor_2']['Layer'][1]),
#             nn.ReLU(),
#         )
#
#         self.attention = nn.Sequential(
#             nn.Linear(self.L, self.D),
#             nn.Tanh(),
#             nn.Linear(self.D, self.K)
#         )
#
#         self.classifier = nn.Sequential(
#             nn.Linear(self.L * self.K, 1),
#             # nn.Sigmoid()
#         )
#
#     def forward(self, x):
#
#         r'''
#         Instance method to get modification probability on the site level from instance features.
#
#                 Args:
#                         x (torch.Tensor): A tensor representation of the instance features
#                 Returns:
#                         Y_prob (torch.Tensor): A tensor containing modification probability on the bag level
#                         Y_hat (torch.Tensor): Binary format of Y_prob
#                         A (torch.Tensor): A tensor containing attention weight of instance features
#         '''
#
#         x = x.squeeze(0)
#
#         H = self.feature_extractor_part1(x)
#         H = H.view(-1, 50 * 4 * 4)
#         H = self.feature_extractor_part2(H)  # NxL
#
#         A = self.attention(H)  # NxK
#         A = torch.transpose(A, 1, 0)  # KxN
#         A = F.softmax(A, dim=1)  # softmax over N
#
#         M = torch.mm(A, H)  # KxL
#
#         Y_prob = self.classifier(M)
#         Y_hat = torch.ge(Y_prob, 0.5).float()
#
#         return Y_prob, Y_hat, A
#
#     def bag_forward(self, input):
#
#         r'''
#            Instance method to get modification probability on the bag level from instance features.
#
#                    Args:
#                            input (torch.Tensor, torch.Tensor): Tensor representation of the bag features and bag labels
#                    Returns:
#                            Y_prob (torch.Tensor): A tensor containing modification probability on the bag level
#                            Y_hat (torch.Tensor): Binary format of Y_prob
#                            A (torch.Tensor): A tensor containing attention weight of instance features
#            '''
#
#
#         bag, bag_labels = input
#
#         bag_pro = [self.forward(item) for item in bag]
