from sklearn.ensemble import RandomForestRegressor
from typing import *

class RF(object):

    """
    class to bulid random forest model
    """

    def __init__(self, model_config: Dict):

        r"""
        Initialization function for the class

            Args:
                    model_config (Dict): A dictionary containing model_factory configurations.

            Returns:
                    None
        """

        super(RF, self).__init__()

        self.model_config = model_config
        self.build_model()

    def build_model(self):

        r'''
        Instance method for building random forest model according to config
        '''

        "1.Build hyper-parameter"
        # self.random_grid = self.model_config['randomforest']

        "2.Build RF model"
        self.rf = RandomForestRegressor()