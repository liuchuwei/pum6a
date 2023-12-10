from model_factory.factory import *
import torch

class puma(nn.Module):

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

        super(puma, self).__init__()

        self.device = (
            "cuda"
            if torch.cuda.is_available() and model_config['device'] == 'cuda'
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        self.model_config = model_config
        self.build_model()

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

    def build_AE(self):

        r'''
        Instance method for building autoencoder module according to config
        '''
        AE = AutoEncoder(self.model_config)
        self.encoder = AE.encoder.to(self.device)
        self.decoder = AE.decoder.to(self.device)

    def build_model(self):

        r'''
        Instance method for building pum6a model_factory according to config
        '''

        '1. build autoencoder'
        self.build_AE()

        '2. build logistic module'
        self.build_logistic()

    def _forward(self, x):

        r'''
        Instance method to get modification probability on the site level from instance features.

                Args:
                        x (torch.Tensor): A tensor representation of the instance features
                Returns:
                        Y_prob (torch.Tensor): A tensor representation the bag probability
                        A (torch.Tensor): A tensor containing attention weight of instance features

        '''

        enc = self.encoder(x)
        dec = self.decoder(enc)
        l1 = torch.nn.PairwiseDistance(p=2)(x, dec)

        pij = self._logistic(l1)
        pij = pij.unsqueeze(0)

        Y_prob = weightnoisyor(pij)

        return Y_prob, pij

    def bag_forward(self, input):

        r'''
        Instance method to get modification probability on the bag level from instance features.

               Args:
                       input (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
                       Tensor representation of bag, bag_labels, and number of instance
               Returns:
                       loss (torch.Tensor): A tensor representation of model_factory loss

       '''


        bag, bag_labels, n_instance = input

        idx_l1 = torch.where(bag_labels != 1)[0]
        if idx_l1.shape[0] > 0:
            data_inst_l1 = torch.concat([bag[item] for item in idx_l1])
            enc = self.encoder(data_inst_l1)
            dec = self.decoder(enc)
            l1 = torch.nn.PairwiseDistance(p=2)(data_inst_l1, dec)
            data_inst_l1 = data_inst_l1[torch.where(l1 < torch.quantile(l1, 0.95))]
            enc = self.encoder(data_inst_l1)
            dec = self.decoder(enc)
            loss = torch.nn.MSELoss()(data_inst_l1,dec)
        else:
            loss = 0.01*(self.A**2+self.B**2)[0]

        data_inst = [self._forward(item) for item in bag]
        idx_l1 = torch.where(bag_labels != 0)[0]
        if idx_l1.shape[0] > 0:
            l2 = [data_inst[item] for item in idx_l1]
            pi = torch.stack([item[0] for item in l2])
            y = torch.stack([bag_labels[index] for index in idx_l1])
            loss += -1 * (log_diverse_density(pi, y) + 1e-10)
        else:
            loss += 0.01*(self.A**2+self.B**2)[0]

        return loss, data_inst

    def validation(self, bag, bag_label):

        """
        instance method to get bag probability and likihood loss

            Args:
                bag (list): bag dataset
                bag_label (list): bag label
            Return:
                bag_loss (torch.Tensor): bag likelihood loss
        """

        data_inst = [self._forward(item.to(self.device)) for item in bag]
        bag_pro = torch.concat([item[0] for item in data_inst]).to(self.device)
        bag_loss = torch.sum(-1. * (bag_label * torch.log(bag_pro) + (1. - bag_label) * torch.log(1. - bag_pro)))
        inf = float("inf")
        if bag_loss == inf:
            bag_loss = 6666666

        if torch.isnan(bag_loss):
            bag_loss = 6666666

        return bag_loss

    def decision(self, bag):

        """
        instance method to get bag probability and instance probability

            Args:
                bag (list): bag dataset
            Return:
                bag_pro (torch.Tensor): tensor representation of bag probability
                ins_pro (torch.Tensor): tensor representation of instance probability
        """

        data_inst = [self._forward(item.to(self.device)) for item in bag]
        bag_pro = torch.concat([item[0] for item in data_inst]).to(self.device)
        ins_pro = torch.concat([item[1].squeeze() for item in data_inst]).to(self.device)

        return bag_pro, ins_pro