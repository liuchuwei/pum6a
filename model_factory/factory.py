from typing import *
import torch
import torch.nn as nn


def GetActivation(name):
    r'''
    Instance method for building activation function

        Args:
            name (str): activation name. possible value: relu、sigmoid、 tanh

        Return:
            none
    '''

    activations = {
        'relu': torch.nn.ReLU(),
        'sigmoid': torch.nn.Sigmoid(),
        'tanh': torch.nn.Tanh(),
        'softmax': torch.nn.Softmax(),
    }

    if name in activations.keys():
        return activations[name]

    else:
        raise ValueError(name, "is not a valid activation function")



def build_attention(L: Optional[int]=500,
                    D: Optional[int]=128,
                    K: Optional[int]=1):

    r'''
    Instance method for building attention module according

        Args:
            L (int):  attention module input nodes
            D (int):  attention module intermediate nodes
            K (int):  attention module output nodes

        Return:
            attention (torch.nn.Sequential): attention module
    '''

    attention = nn.Sequential(
        nn.Linear(L, D),
        nn.Tanh(),
        nn.Linear(D, K)
    )

    return attention

def build_classifier(input: Optional[int]=500):

    r'''
    Instance method for building classifier module according

        Args:
            input (int):  attention module input nodes

        Return:
            classifier (torch.nn.Sequential): classifier module
    '''


    classifier = nn.Sequential(
        nn.Linear(input, 1),
        nn.Sigmoid()
    )

    return classifier


def build_logistic(device: Optional[str]='cuda'):

    '''

    Instance method for building logistic module

        Args:
            device (str): whether to use cuda for calculation

        Return:
            A (torch.nn.Parameter): weight parameter of logistic module
            B (torch.nn.Parameter): bias parameter of logistic module

    '''

    A = torch.nn.Parameter(torch.tensor(torch.rand(1, device=device)), requires_grad=True)
    A.grad = torch.tensor(torch.rand(1, device=device))
    B = torch.nn.Parameter(torch.tensor(torch.rand(1, device=device)), requires_grad=True)

    return A, B

def weightnoisyor(pij: Optional[list] = None,
                  device: Optional[str] = 'cuda',
                  mu1: Optional[int] = 0,
                  mu2: Optional[int] = 1,
                  sigma1: Optional[int] = 0.1,
                  sigma2: Optional[int] = 0.1,
                  ):

    """
    instance method to calculate bag probability based on instance probability

        Args:
            pij (torch.Tensor): Tensor representation of instance probability
            device (str): Whether to use gpu for calculation
            mu1 (int): loc of torch normal distributions
            mu2 (int): loc of torch normal distributions
            sigma1 (int): scale of torch normal distributions
            sigma1 (int): scale of torch normal distributions
        Return:
            noisyor (torch.Tensor): Tensor representation of bag probability
    """

    rv1 = torch.distributions.normal.Normal(loc=torch.tensor(mu1), scale=torch.tensor(sigma1))
    rv2 = torch.distributions.normal.Normal(loc=torch.tensor(mu2), scale=torch.tensor(sigma2))
    nbags = 1
    ninstances = pij.size()[1]
    pij = pij.reshape(nbags, ninstances)
    ranks = torch.empty((nbags, ninstances), dtype=torch.float)
    tmp = torch.argsort(pij, dim=1, descending=False)
    for i in range(nbags):
        ranks[i,tmp[i,:]] = torch.arange(0,ninstances)/(ninstances-1)
    w = torch.exp(rv1.log_prob(ranks))+torch.exp(rv2.log_prob(ranks))
    w = torch.div(w,torch.sum(w, dim=1).reshape(nbags,1))
    pij = pij.to(device, non_blocking = True).float()
    w = w.to(device, non_blocking = True).float()
    noisyor = 1 - torch.prod(torch.pow(1-pij+1e-10,w).clip(min=0, max=1), dim=1)

    return noisyor

class FeatureExtractor(object):

    r"""
    Object class for building linear or conv feature extractor
    """

    def __init__(self, model_config: Dict):

        r"""
        Initialization function for the class

            Args:
                    model_config (Dict): A dictionary containing model_factory configurations.

            Returns:
                    None
        """

        self.model_config = model_config
        self.build_FE()

    def build_linearFE(self):

        self.feature_extractor = nn.Sequential()

        self.n_features = self.model_config['feature_extractor']['n_features']
        self.dropout_rate = self.model_config['feature_extractor']['dropout_rate']
        self.batch_norm = self.model_config['feature_extractor']['batch_norm']
        self.hidden_activation = self.model_config['feature_extractor']['hidden_activation']
        hidden_neurons = self.model_config['feature_extractor']['hidden_neurons']
        self.layers_neurons_ = [self.n_features, *hidden_neurons]

        for idx, layer in enumerate(self.layers_neurons_[:-1]):
            if self.batch_norm:
                self.feature_extractor.add_module("batch_norm" + str(idx), torch.nn.BatchNorm1d(self.layers_neurons_[idx]))
            self.feature_extractor.add_module("linear" + str(idx),
                                    torch.nn.Linear(self.layers_neurons_[idx], self.layers_neurons_[idx + 1]))
            self.feature_extractor.add_module(self.hidden_activation + str(idx), GetActivation(self.hidden_activation))
            self.feature_extractor.add_module("dropout" + str(idx), torch.nn.Dropout(self.dropout_rate))

    def build_convFE(self):

        r'''
        Instance method for building conv feature extractor module according to config
        '''

        self.feature_extractor_1 = nn.Sequential()
        self.feature_extractor_2 = nn.Sequential()

        hidden_neurons = self.model_config['feature_extractor']['hidden_neurons']
        self.layers_neurons_ = [1, *hidden_neurons]
        self.kernal_size = self.model_config['feature_extractor']['kernal_size']
        self.pool_kernal_size = self.model_config['feature_extractor']['pool_kernal_size']
        self.pool_stride = self.model_config['feature_extractor']['pool_stride']
        self.hidden_activation = self.model_config['feature_extractor']['hidden_activation']

        for idx, layer in enumerate(self.layers_neurons_[:-1]):

            self.feature_extractor_1.add_module("conv" + str(idx),
                nn.Conv2d(self.layers_neurons_[idx], self.layers_neurons_[idx + 1], kernel_size=self.kernal_size))
            self.feature_extractor_1.add_module("hidden_activation" + str(idx), GetActivation(self.hidden_activation))
            self.feature_extractor_1.add_module("pool" + str(idx),
                                    nn.MaxPool2d(self.pool_kernal_size, stride=self.pool_stride))

        self.L = self.model_config['attention']['L']
        self.linear_input = self.model_config['feature_extractor']['linear_input']
        self.feature_extractor_2.add_module("linear",
                                            torch.nn.Linear(self.linear_input, self.L))
        self.feature_extractor_2.add_module("hidden_activation", GetActivation(self.hidden_activation))

    def build_FE(self):

        r'''
        Instance method for building feature extractor module according to config
        '''

        self.FE_type = self.model_config['feature_extractor']['type']
        if self.FE_type == 'linear':

            self.build_linearFE()

        elif self.FE_type == 'conv':

            self.build_convFE()

        else:

            raise ValueError('invalid feature extractor type!')