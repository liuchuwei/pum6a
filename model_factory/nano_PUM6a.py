from model_factory.factory import *
import torch.nn.functional as F
import torch
import numpy as np

class Nanopum6a(nn.Module):

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

        super(Nanopum6a, self).__init__()

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

    def build_attention(self):

        r'''
        Instance method for building attention module according to config
        '''


        self.attention = build_attention(
            L=self.model_config['attention']['L'],
            D=self.model_config['attention']['D'],
            K=self.model_config['attention']['K'],
        ).to(self.device)

    def build_classifier(self):

        r'''
        Instance method for building classifier module according to config
        '''


        self.classifier = build_classifier(
            input=self.model_config['attention']['L'] * self.model_config['attention']['K']
        ).to(self.device)

    def build_logistic(self):

        r'''
        Instance method for building logistic module according to config
        '''

        self.A, self.B = build_logistic(device=self.device)

    def _logistic(self, loss):

        r"""
        instance method to get instance probability according to reconstruction loss

            Args:
                loss (torch.Tensor): reconstruction loss
            Return:
                (torch.Tensor): instance probability
        """
        return torch.sigmoid(self.A * loss + self.B)

    def build_model(self):

        r'''
        Instance method for building pum6a model according to config
        '''

        '1. build feature extractor'
        self.build_FE()

        '2. build attention module'
        self.build_attention()

        '3. build classifier'
        self.build_classifier()

        '4. build logistic module'
        self.build_logistic()

    def Attforward(self, x):

        r'''
        Instance method to get modification probability on the site level from instance features.

                Args:
                        x (torch.Tensor): A tensor representation of the instance features
                Returns:
                        Y_prob (torch.Tensor): A tensor representation the bag probability
                        A (torch.Tensor): A tensor containing attention weight of instance features
                        pi (torch.Tensor): A tensor representation the bag probability
                        pij (torch.Tensor): A tensor containing attention weight of instance features

        '''

        if self.FE_type == "linear":
            H = self.feature_extractor(x)

        elif self.FE_type == "conv":
            H = self.feature_extractor_1(x)
            H = H.view(-1, 50 * 4 * 4)
            H = self.feature_extractor_2(H)  # NxL

        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        pij = self._logistic(A)
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)

        if pij.size()[1]>1:
            pi = weightnoisyor(pij)
        else:
            pi = 1 - torch.prod(1-pij, dim = 1)


        return pi, pij, Y_prob, A

    def bag_forward(self, input):

        r'''
        Instance method to get modification probability on the bag level from instance features.

               Args:
                       input (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
                       Tensor representation of bag, bag_labels, and number of instance
               Returns:
                       loss (torch.Tensor): A tensor representation of model_factory loss
                       data_inst (torch.Tensor...): pi, pij, Y_prob, A
       '''


        bag, bag_labels, n_instance = input

        data_inst = [self.Attforward(item) for item in bag]
        idx_l2 = torch.where(bag_labels != 0)[0]
        if idx_l2.shape[0] > 0:
            l2 = [data_inst[item] for item in idx_l2]
            p = torch.stack([item[2] for item in l2])
            pi= torch.stack([item[0] for item in l2])
            y = torch.stack([bag_labels[index] for index in idx_l2])
            y = y.to(self.device)
            p = p.to(self.device)
            pi = pi.to(self.device)
            loss = -1*(log_diverse_density(pi, y)+1e-10) + 0.01*(self.A**2+self.B**2)[0]
            y[torch.where(y == -1)] = 0
            loss += torch.sum(-1. * (y * torch.log(p) + (1. - y) * torch.log(1. - p)))  # pro
        else:
            loss = 0.01*(self.A**2+self.B**2)[0]

        return loss, data_inst

    def validation(self, bag):

        """
        instance method to get bag probability and likihood loss

            Args:
                bag: bag dataset
            Return:
                bag_loss (torch.Tensor): bag likelihood loss
        """

        bags = []
        id = bag.keys
        for item in id:
            reads = bag.sitedict[item]
            feature = np.stack([bag.feature[item, :] for item in reads])
            feature = torch.tensor(feature, dtype=torch.float32)
            bags.append(feature)

        data_inst = [self.Attforward(item.to(self.device)) for item in bags]
        bag_pro = torch.concat([item[0] for item in data_inst]).to(self.device)

        bag_label = bag.keys_mod.to(self.device)
        bag_loss = torch.sum(-1. * (bag_label * torch.log(bag_pro) + (1. - bag_label) * torch.log(1. - bag_pro)))

        return bag_loss

    def decision(self, bag):

        """
        instance method to get bag probability and instance probability

            Args:
                bag (list): bag dataset
            Return:
                bag_pro (torch.Tensor): tensor representation of bag probability
        """

        bags = []
        id = bag.keys
        for item in id:
            reads = bag.sitedict[item]
            feature = np.stack([bag.feature[item, :] for item in reads])
            feature = torch.tensor(feature, dtype=torch.float32)
            bags.append(feature)

        data_inst = [self.Attforward(item.to(self.device)) for item in bags]
        bag_pro = torch.concat([item[0] for item in data_inst]).to(self.device)

        bag_label = bag.keys_mod.to(self.device)

        return bag_pro, bag_label