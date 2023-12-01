import torch.nn as nn
from typing import *
import torch
import torch.nn.functional as F


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

        pass

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

    def build_logistic(self):

        r'''
        Instance method for building logistic module according to config
        '''

        self.A = torch.nn.Parameter(torch.tensor(torch.rand(1)), requires_grad=True)
        self.A.grad = torch.tensor(torch.rand(1))
        self.B = torch.nn.Parameter(torch.tensor(torch.rand(1)), requires_grad=True)


    def build_model(self):

        r'''
        Instance method for building pum6a model according to config
        '''

        '1. build feature extractor'
        self.build_FE()

        '2. build attention module'
        self.build_attention()

        '3. build logistic module'
        self.build_logistic()


    def Attforward(self, x):

        r'''
        Instance method to get modification probability on the site level from instance features.

                Args:
                        x (torch.Tensor): A tensor representation of the instance features
                Returns:
                        Y_prob (torch.Tensor): A tensor representation the bag probability
                        A (torch.Tensor): A tensor containing attention weight of instance features

        '''
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, 50 * 4 * 4)
        H = self.feature_extractor_part2(H)  # NxL

        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)

        return Y_prob, A


    def bag_forward(self, input):

        r'''
        Instance method to get modification probability on the bag level from instance features.

               Args:
                       input (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
                       Tensor representation of bag, bag_labels, and number of instance
               Returns:
                       loss (torch.Tensor): A tensor representation of model loss

       '''


        bag, bag_labels, n_instance = input

        data_inst = [self.Attforward(item) for item in bag]
        idx_l2 = torch.where(bag_labels != 0)[0]
        if idx_l2.shape[0] > 0:
            l2 = [data_inst[item] for item in idx_l2]
            pi = torch.stack([item[0] for item in l2])
            y = torch.stack([bag_labels[index] for index in idx_l2])
            y[torch.where(y == -1)] = 0
            loss = torch.sum(-1. * (y * torch.log(pi) + (1. - y) * torch.log(1. - pi)))
        else:
            loss = 0.01*(self.A**2+self.B**2)[0]

        return loss, data_inst


#
# class pum6a(nn.Module):
#
#     r"""
#     The positive and unlabeled multi-instance model.
#
#     """
#     def __init__(self, model_config: Dict):
#
#         r"""
#         Initialization function for the class
#
#             Args:
#                     model_config (Dict): A dictionary containing model configurations.
#
#             Returns:
#                     None
#         """
#
#         super(pum6a, self).__init__()
#
#         self.model_config = model_config
#         self.build_model()
#
#         self.device = (
#             "cuda"
#             if torch.cuda.is_available() and model_config['device'] == 'cuda'
#             else "mps"
#             if torch.backends.mps.is_available()
#             else "cpu"
#         )
#
#     def build_model(self):
#
#         r'''
#         Instance method for building pum6a model according to config
#         '''
#
#         self.L = self.model_config['attention']['L']
#         self.D = self.model_config['attention']['D']
#         self.K = self.model_config['attention']['K']
#
#
#         self.feature_extractor_part1 = nn.Sequential(
#             nn.Conv2d(1, 20, kernel_size=5),
#             nn.ReLU(),
#             nn.MaxPool2d(2, stride=2),
#             nn.Conv2d(20, 50, kernel_size=5),
#             nn.ReLU(),
#             nn.MaxPool2d(2, stride=2)
#         )
#
#         self.feature_extractor_part2 = nn.Sequential(
#             nn.Linear(50 * 4 * 4, self.L),
#             nn.ReLU(),
#         )
#
#         self.attention = nn.Sequential(
#             nn.Linear(self.L, self.D),
#             nn.Tanh(),
#             nn.Linear(self.D, self.K)
#         )
#
#         self.classifier = nn.Sequential(
#             nn.Linear(self.L * self.K, 1),
#             nn.Sigmoid()
#         )
#
#         self.A = torch.nn.Parameter(torch.tensor(torch.rand(1)), requires_grad=True)
#         self.B = torch.nn.Parameter(torch.tensor(torch.rand(1)), requires_grad=True)
#
#     def Attforward(self, x):
#
#         r'''
#         Instance method to get modification probability on the site level from instance features.
#
#                 Args:
#                         x (torch.Tensor): A tensor representation of the instance features
#                 Returns:
#                         Y_prob (torch.Tensor): A tensor representation the bag probability
#                         A (torch.Tensor): A tensor containing attention weight of instance features
#
#         '''
#         x = x.squeeze(0)
#
#         H = self.feature_extractor_part1(x)
#         H = H.view(-1, 50 * 4 * 4)
#         H = self.feature_extractor_part2(H)  # NxL
#
#         A = self.attention(H)  # NxK
#         A = torch.transpose(A, 1, 0)  # KxN
#         A = F.softmax(A, dim=1)  # softmax over N
#
#         M = torch.mm(A, H)  # KxL
#
#         Y_prob = self.classifier(M)
#
#         return Y_prob, A
#
#
#     def bag_forward(self, input):
#
#         r'''
#         Instance method to get modification probability on the bag level from instance features.
#
#                Args:
#                        input (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
#                        Tensor representation of bag, bag_labels, and number of instance
#                Returns:
#                        loss (torch.Tensor): A tensor representation of model loss
#
#        '''
#
#
#         bag, bag_labels, n_instance = input
#
#         data_inst = [self.Attforward(item) for item in bag]
#         idx_l2 = torch.where(bag_labels != 0)[0]
#         if idx_l2.shape[0] > 0:
#             l2 = [data_inst[item] for item in idx_l2]
#             pi = torch.stack([item[0] for item in l2])
#             y = torch.stack([bag_labels[index] for index in idx_l2])
#             y[torch.where(y == -1)] = 0
#             loss = torch.sum(-1. * (y * torch.log(pi) + (1. - y) * torch.log(1. - pi)))
#         else:
#             loss = 0.01*(self.A**2+self.B**2)[0]
#
#         return loss, data_inst


class pum6a_1(nn.Module):

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

        super(pum6a_1, self).__init__()

        self.model_config = model_config
        self.build_model()

        self.device = (
            "cuda"
            if torch.cuda.is_available() and model_config['device'] == 'cuda'
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    def get_activation_by_name(self, name):

        r'''
        Instance method for building activation function

            Args:
                name: activation name. possible value: relu、sigmoid、 tanh

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
        self.layers_neurons_decoder_ = self.layers_neurons_[::-1]
        self.latent = self.model_config['autoencoder']['latent']

        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()

        for idx, layer in enumerate(self.layers_neurons_[:-1]):

            self.encoder.add_module("conv" + str(idx),
                nn.Conv2d(self.layers_neurons_[idx], self.layers_neurons_[idx + 1], kernel_size=self.kernal_size))
            self.encoder.add_module("hidden_activation" + str(idx), self.hidden_activation)


        self.encoder.add_module("faltten",nn.Flatten())
        n_out_feat = (self.n_features[0] - len(hidden_neurons)*(self.kernal_size-1))
        n_linear = n_out_feat**2*hidden_neurons[-1]
        self.encoder.add_module("linear_last", nn.Linear(n_linear,self.latent))
        self.encoder.add_module("out_acitvation",self.encoder_activation)

        self.decoder.add_module("linear", nn.Linear(self.latent, n_linear))
        self.decoder.add_module("input_activation", self.hidden_activation)
        self.decoder.add_module("unflatten", nn.Unflatten(1, (self.layers_neurons_decoder_[0], n_out_feat, n_out_feat)))

        for idx, layer in enumerate(self.layers_neurons_decoder_[:-1]):

            self.decoder.add_module("convT" + str(idx),
                nn.ConvTranspose2d(self.layers_neurons_decoder_[idx], self.layers_neurons_decoder_[idx + 1], kernel_size=self.kernal_size))
            self.decoder.add_module("hidden_activation" + str(idx), self.hidden_activation)

        self.encoder.add_module("out_acitvation",self.decoder_activation)


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

    def _logistic(self, loss):

        r"""
        instance method to get instance probability according to reconstruction loss

            Args:
                loss (torch.Tensor): reconstruction loss
            Return:
                (torch.Tensor): instance probability
        """
        return torch.sigmoid(self.A * loss + self.B)

    def autoencoder_forward(self, x):

        r"""
            Instance method to get modification probability on the site level from instance features.

                Args:
                    x (torch.Tensor): A tensor representation of the instance features

                Returns:
                    enc (torch.Tensor): A tensor representation of the autoencoder latent features
                    dec (torch.Tensor): A tensor representation of the autoencoder reconstruct features
        """

        enc = self.encoder(x)
        dec = self.decoder(enc)

        return enc, dec

    def _weightnoisyor(self,pij):

        """
        instance method to calculate instance weight
        """
        self.mu1 = 0
        self.mu2 = 1
        self.sigma1 = 0.1
        self.sigma2 = 0.1

        rv1 = torch.distributions.normal.Normal(loc=torch.tensor(self.mu1), scale=torch.tensor(self.sigma1))
        rv2 = torch.distributions.normal.Normal(loc=torch.tensor(self.mu2), scale=torch.tensor(self.sigma2))
        # nbags = pij.size()[0]
        nbags = 1
        # ninstances = pij.size()[1]
        ninstances = pij.size()[0]
        pij = pij.reshape(nbags,ninstances)
        ranks = torch.empty((nbags, ninstances), dtype = torch.float)
        tmp = torch.argsort(pij, dim=1, descending=False)
        for i in range(nbags):
            ranks[i,tmp[i,:]] = torch.arange(0,ninstances)/(ninstances-1)
        w = torch.exp(rv1.log_prob(ranks))+torch.exp(rv2.log_prob(ranks))
        w = torch.div(w,torch.sum(w, dim = 1).reshape(nbags,1))
        pij = pij.to(self.device, non_blocking = True).float()
        w = w.to(self.device, non_blocking = True).float()
        noisyor = 1 - torch.prod(torch.pow(1-pij+1e-10,w).clip(min = 0, max = 1), dim = 1)
        return noisyor

    def Attforward(self, x):

        r'''
        Instance method to get modification probability on the site level from instance features.

                Args:
                        x (torch.Tensor): A tensor representation of the instance features
                Returns:
                        bp (torch.Tensor): A tensor representation the bag probability
                        pij (torch.Tensor): A tensor representation the instance probability
                        A (torch.Tensor): A tensor containing attention weight of instance features

        '''
        x = x.view(x.size(0), -1)

        enc, dec = self.autoencoder_forward(x)

        if len(x.size()) >= 2:
            l1 = torch.nn.PairwiseDistance(p=2)(x.flatten(start_dim=1), dec.flatten(start_dim=1))
        else:
            l1 = torch.nn.PairwiseDistance(p=2)(x, dec)

        # l1 = torch.stack([(1-ssim(x[idx].unsqueeze(0), dec[idx].unsqueeze(0))) for idx in range(len(x))])
        pij = self._logistic(l1)
        #
        # H = self.feature_extractor_part1(x)
        # H = H.view(-1, 50 * 4 * 4)
        # H = self.feature_extractor_part2(H)  # NxL

        A = self.attention(pij.unsqueeze(1))  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        # M = torch.mm(A, enc)
        bc = torch.mm(A, pij.unsqueeze(1))
        # bc = self.classifier(bc)

        # pij = torch.mul(pij, A.squeeze())
        # w = self.a_to_w(A)
        bp = 1 - torch.prod(torch.pow(1 - pij + 1e-10, A).clip(min=0, max=1), dim=1)
        # bp = (1 - torch.prod((1-pij+1e-10).clip(min=0, max=1))).unsqueeze(0)
        # bp = self._weightnoisyor(pij)

        return bc, bp, pij, A

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
                       loss (torch.Tensor): A tensor representation of model loss

       '''


        bag, bag_labels, n_instance = input

        idx_l1 = torch.where(bag_labels != 1)[0]

        if idx_l1.shape[0] > 0:
            data_inst_l1 = torch.concat([bag[index] for index in idx_l1])
            data_inst_l1 = data_inst_l1.view(data_inst_l1.size(0), -1)
            enc, dec = self.autoencoder_forward(data_inst_l1)
            if len(data_inst_l1.size())>=2:
                l1 = torch.nn.PairwiseDistance(p=2)(data_inst_l1.flatten(start_dim=1), dec.flatten(start_dim=1))
            else:
                l1 = torch.nn.PairwiseDistance(p=2)(data_inst_l1, dec)
            data_inst_l1 = data_inst_l1[torch.where(l1 < torch.quantile(l1, 1 - self.cont_factor, dim=0))]
            enc, dec = self.autoencoder_forward(data_inst_l1)
            loss1 = torch.nn.MSELoss()(data_inst_l1, dec)
        else:
            loss1 = torch.tensor(0, dtype=torch.float) .to(self.device) # reconstruct loss

        data_inst = [self.Attforward(item) for item in bag]
        # data_inst = self.Attforward(torch.concat(bag), n_instance=n_instance)
        idx_l2 = torch.where(bag_labels != 0)[0]
        if idx_l2.shape[0] > 0:
            l2 = [data_inst[item] for item in idx_l2]
            pc = torch.concat([item[0] for item in l2])
            # pc = 0
            pi = torch.stack([item[1] for item in l2])
            # pi = data_inst[1][idx_l2]
            y = torch.stack([bag_labels[index] for index in idx_l2])
            # yi = torch.clone(y)
            # yi[torch.where(yi == -1)] = 0
            loss2 = -1*(self._log_diverse_density(pi, y)+1e-10) + 0.01*(self.A**2+self.B**2)[0]
            # loss3 = torch.sum(-1. * (yi * torch.log(pc) + (1. - yi) * torch.log(1. - pc)))
            loss3 = -1*(self._log_diverse_density(pc, y)+1e-10) + 0.01*(self.A**2+self.B**2)[0]
            # loss3 *= 0.1
            # loss3 = torch.tensor(0, dtype=torch.float).to(self.device)

        else:
            loss2 = 0.01*(self.A**2+self.B**2)[0]
            loss3 = torch.tensor(0, dtype=torch.float).to(self.device)

        # bag_pi = torch.stack([item[0] for item in data_inst])

        return loss1, loss2, loss3, data_inst
