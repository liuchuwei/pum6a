import torch.nn as nn
from typing import *
import toml


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


    def forward(self):
        pass