from model_factory.factory import *
import torch.nn.functional as F

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

        '3. build classifier module'
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
        x = x.squeeze(0)

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

        pi = weightnoisyor(pij)

        return pi, pij, Y_prob, A

    def _log_diverse_density(self, pi, y_bags):
        r'''
        Instance method to Compute the likelihood given bag labels y_bags and bag probabilities pi.
                Args:
                        pi (torch.Tensor): A tensor representation of the bag probabilities
                        y_bags (torch.Tensor): A tensor representation of the bag labels
                Returns:
                        likelihood (torch.Tensor): A tensor representation of the likelihood

        '''

        z = torch.where(y_bags == -1)[0]
        if z.nelement() > 0:
            zero_sum = torch.sum(torch.log(1 - pi[z] + 1e-10))
        else:
            zero_sum = torch.tensor(0).float()

        o = torch.where(y_bags == 1)[0]
        if o.nelement() > 0:
            one_sum = torch.sum(torch.log(pi[o] + 1e-10))
        else:
            one_sum = torch.tensor(0).float()
        return zero_sum + one_sum

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
            loss = -1*(self._log_diverse_density(pi, y)+1e-10) + 0.01*(self.A**2+self.B**2)[0]
            y[torch.where(y == -1)] = 0
            loss += torch.sum(-1. * (y * torch.log(p) + (1. - y) * torch.log(1. - p)))  # pro
        else:
            loss = 0.01*(self.A**2+self.B**2)[0]

        return loss, data_inst