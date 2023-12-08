from model_factory.factory import *
import torch
from sklearn.ensemble import IsolationForest

class puIF(torch.nn.Module):

    r"""
    The puIF model.
    """

    def __init__(self, model_config: Dict):
        r"""
                Initialization function for the class

                    Args:
                            model_config (Dict): A dictionary containing model_factory configurations.

                    Returns:
                            None
                """

        super(puIF, self).__init__()

        self.device = (
            "cuda"
            if torch.cuda.is_available() and model_config['device'] == 'cuda'
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        self.model_config = model_config
        self.build_model()


    def build_FE(self):

        r'''
        Instance method for building feature extractor module according to config
        '''

        FE = FeatureExtractor(self.model_config)
        if self.model_config['feature_extractor']['type']=='conv':

            self.feature_extractor_1 = FE.feature_extractor_1.to(self.device)
            self.feature_extractor_2 = FE.feature_extractor_2.to(self.device)

        elif self.model_config['feature_extractor']['type']=='linear':
            self.feature_extractor = FE.feature_extractor.to(self.device)

        self.FE_type = self.model_config['feature_extractor']['type']

    def build_classifier(self):

        r'''
        Instance method for building classifier module according to config
        '''


        self.classifier = build_classifier(
            input=self.model_config['classifier']['input']
        ).to(self.device)

    def build_model(self):

        """
        Instance method to build model
        """

        "1. build isolation forest model_factory"
        self.clf = IsolationForest(contamination=self.model_config['isoForest']['contamination'],
                                   random_state=888)

        '2. build feature extractor'
        self.build_FE()

        '3. build bias parameter'
        self.B = torch.nn.Parameter(torch.tensor(torch.rand(1), device=self.device), requires_grad=True)

    def bag_forward(self, input):

        r'''
        Instance method to get modification probability on the bag level from instance features.

               Args:
                       input (torch.Tensor, torch.Tensor):Tensor representation of instance and instance probability
               Returns:
                       loss (torch.Tensor): A tensor representation of model loss
        '''

        X, y = input
        lin = self.feature_extractor(X.to(self.device))
        ps = 1 / (1 + self.B ** 2 + torch.exp(-lin))
        y[torch.where(y == 1)] = 0
        y[torch.where(y == -1)] = 1
        y = y.to(self.device)
        loss = torch.sum(-1. * (y * torch.log(ps) + (1. - y) * torch.log(1. - ps)))

        return loss

    def decision(self, bag):

        """
        Instance method to obtain bag probability and instance probability
        """
        ins_pro = []
        bag_pro = []

        for item in bag:

            lin = self.feature_extractor(item.to(self.device))
            ps = 1 / (1 + self.B ** 2 + torch.exp(-lin))
            c_hat = 1 / (1 + self.B ** 2)
            p = ps / c_hat
            p = p.T
            ins_pro.append(p)
            bag_pro.append(weightnoisyor(p))

        return ins_pro, bag_pro

