import torch
import torch.nn as nn
from typing import *

class pum6a(nn.Module):

    r"""
    The Attention-based Positive and Unlabeled Multi-instance model.
    """

    def __init__(self, model_config: Dict):

        r"""
        Initialization function for the class

            Args:
                    model_config (Dict): A dictionary containing model_factory configurations.

            Returns:
                    None
        """

        super(pum6a, self).__init__()

        self.device = (
            "cuda"
            if torch.cuda.is_available() and model_config['device'] == 'cuda'
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        self.model_config = model_config
        self.build_model()

    def build_model(self):

        r'''
        Instance method for building pum6a model according to config
        '''

        self.L = self.model_config['attention']['L']
        self.D = self.model_config['attention']['D']
        self.K = self.model_config['attention']['K']

        '1. build feature extractor'
        self.build_FE()

        '2. build attention module'
        self.build_attention()

        '3. build classifier module'
        self.build_classifier()

        '4. build logistic module'
        self.build_logistic()