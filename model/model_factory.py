import torch.nn as nn
from typing import *
import torch

class pum6a(nn.Module):

    r"""
    The positive and unlabeled multi-instance model.

    """
    def __init__(self, model_config: Dict):

        r"""
        Initialization function for the class

            Args:
                    model_config (Dict): A dictionary containing model configurations.

            Returns:
                    None
        """

        super(pum6a, self).__init__()

        self.model_config = model_config
        self.build_model()

    def get_activation_by_name(self, name):

        r'''
        Instance method for building activation function

            Args:
                name: activation name. possible value: relu、sigmoid、 tanh

            Return:
                none
        '''
        activations = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'softmax': nn.Softmax(),
        }

        if name in activations.keys():
            return activations[name]

        else:
            raise ValueError(name, "is not a valid activation function")

    def build_linearAE(self):

        r'''
        Instance method for building linear autoencoder module according to config
        '''

        self.n_features = self.model_config['autoencoder']['n_features']
        self.dropout_rate = self.model_config['autoencoder']['dropout_rate']
        self.batch_norm = self.model_config['autoencoder']['batch_norm']
        self.hidden_activation = self.model_config['autoencoder']['hidden_activation']

        self.activation = self.get_activation_by_name(self.hidden_activation)
        hidden_neurons = self.model_config['autoencoder']['hidden_neurons']
        self.layers_neurons_ = [self.n_features, *hidden_neurons]
        self.layers_neurons_decoder_ = self.layers_neurons_[::-1]

        self.encoder = torch.nn.Sequential()
        self.decoder = torch.nn.Sequential()

        for idx, layer in enumerate(self.layers_neurons_[:-1]):
            if self.batch_norm:
                self.encoder.add_module("batch_norm" + str(idx), torch.nn.BatchNorm1d(self.layers_neurons_[idx]))
            self.encoder.add_module("linear" + str(idx),
                                    torch.nn.Linear(self.layers_neurons_[idx], self.layers_neurons_[idx + 1]))
            self.encoder.add_module(self.hidden_activation + str(idx), self.activation)
            self.encoder.add_module("dropout" + str(idx), torch.nn.Dropout(self.dropout_rate))

        for idx, layer in enumerate(self.layers_neurons_[:-1]):
            if self.batch_norm:
                self.decoder.add_module("batch_norm" + str(idx),
                                        torch.nn.BatchNorm1d(self.layers_neurons_decoder_[idx]))
            self.decoder.add_module("linear" + str(idx), torch.nn.Linear(self.layers_neurons_decoder_[idx],
                                                                         self.layers_neurons_decoder_[idx + 1]))
            self.encoder.add_module(self.hidden_activation + str(idx), self.activation)
            self.decoder.add_module("dropout" + str(idx), torch.nn.Dropout(self.dropout_rate))

    def build_convAE(self):

        r'''
        Instance method for building linear autoencoder module according to config
        '''

        self.n_features = self.model_config['autoencoder']['n_features']
        self.kernal_size = self.model_config['autoencoder']['kernal_size']
        self.hidden_activation = self.get_activation_by_name(self.model_config['autoencoder']['hidden_activation'])
        self.encoder_activation = self.get_activation_by_name(self.model_config['autoencoder']['encoder_activation'])
        self.decoder_activation = self.get_activation_by_name(self.model_config['autoencoder']['decoder_activation'])

        hidden_neurons = self.model_config['autoencoder']['hidden_neurons']
        self.layers_neurons_ = [1, *hidden_neurons]
        self.latent = 10

        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()

        for idx, layer in enumerate(self.layers_neurons_[:-1]):

            self.encoder.add_module("conv" + str(idx),
                nn.Conv2d(self.layers_neurons_[idx], self.layers_neurons_[idx + 1], kernel_size=self.kernal_size))
            self.encoder.add_module(self.hidden_activation + str(idx), self.activation)


        self.encoder.add_module(nn.Flatten())
        n_linear = self.n_features
        self.encoder.add_module(nn.linear(n_linear ,self.latent))
        self.encoder.add_module(self.encoder_activation)

    def build_AE(self):

        r'''
        Instance method for building autoencoder module according to config
        '''

        if self.model_config['autoencoder']['type'] == 'linear':

            self.build_linearAE()

        elif self.model_config['autoencoder']['type'] == 'conv':

            self.build_convAE()

        else:

            raise ValueError('invalid autoencoder type!')

    def build_logistic(self):

        r'''
        Instance method for building logistic module according to config
        '''

        self.A = torch.nn.Parameter(torch.tensor(torch.rand(1)), requires_grad=True)
        self.A.grad = torch.tensor(torch.rand(1))
        self.B = torch.nn.Parameter(torch.tensor(torch.rand(1)), requires_grad=True)


    def build_attention(self):

        r'''
        Instance method for building attention module according to config
        '''

        self.L = self.model_config['attention']['L']
        self.D = self.model_config['attention']['D']
        self.K = self.model_config['attention']['K']

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

    def build_model(self):

        r'''
        Instance method for building pum6a model according to config
        '''

        '1. build autoencoder module'
        self.build_AE()

        '2. build logistic module'
        self.build_logistic()

        '3. build attention module'
        self.build_attention()



