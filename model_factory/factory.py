from typing import *
import torch
import torch.nn as nn

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
            self.feature_extractor.add_module(self.hidden_activation + str(idx), self.get_activation_by_name(self.hidden_activation))
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
            self.feature_extractor_1.add_module("hidden_activation" + str(idx), self.get_activation_by_name(self.hidden_activation))
            self.feature_extractor_1.add_module("pool" + str(idx),
                                    nn.MaxPool2d(self.pool_kernal_size, stride=self.pool_stride))


        self.linear_input = self.model_config['feature_extractor']['linear_input']
        self.feature_extractor_2.add_module("linear",
                                            torch.nn.Linear(self.linear_input, self.L))
        self.feature_extractor_2.add_module("hidden_activation", self.get_activation_by_name(self.hidden_activation))

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