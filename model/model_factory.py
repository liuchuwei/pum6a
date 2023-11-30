import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import *
from utils.ssim_metircs import ssim
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

class Net(nn.Module):

    r"""
    The logistic model, use for instance probability inference
    """
    def __init__(self):

        r"""
        Initialization function for the class
        """

        super(Net, self).__init__()

        self.A = torch.nn.Parameter(torch.rand(1))
        self.B = torch.nn.Parameter(torch.rand(1))

    def forward(self, inputs):

        r'''
        Instance method to get instance probability.
        '''

        return torch.sigmoid(self.A * inputs + self.B)


class milpuAtt_construct(nn.Module):

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

        super(milpuAtt_construct, self).__init__()

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
            'tanh': nn.Tanh()
        }

        if name in activations.keys():
            return activations[name]

        else:
            raise ValueError(name, "is not a valid activation function")

    def build_model(self):

        r'''
        Instance method for building milpuAttention model according to config

        '''

        self.L = self.model_config['attention']['L']
        self.D = self.model_config['attention']['D']
        self.K = self.model_config['attention']['K']

        self.encoder = torch.nn.Sequential()
        self.decoder = torch.nn.Sequential()


        self.n_features = self.model_config['autoencoder']['n_features']
        self.dropout_rate = self.model_config['autoencoder']['dropout_rate']
        self.batch_norm = self.model_config['autoencoder']['batch_norm']
        self.hidden_activation = self.model_config['autoencoder']['hidden_activation']

        self.activation = self.get_activation_by_name(self.hidden_activation)
        hidden_neurons=self.model_config['autoencoder']['hidden_neurons']
        self.layers_neurons_ = [self.n_features, *hidden_neurons]
        self.layers_neurons_decoder_ = self.layers_neurons_[::-1]

        for idx, layer in enumerate(self.layers_neurons_[:-1]):
            if self.batch_norm:
                self.encoder.add_module("batch_norm"+str(idx),torch.nn.BatchNorm1d(self.layers_neurons_[idx]))
            self.encoder.add_module("linear"+str(idx),torch.nn.Linear(self.layers_neurons_[idx],self.layers_neurons_[idx+1]))
            self.encoder.add_module(self.hidden_activation+str(idx),self.activation)
            self.encoder.add_module("dropout"+str(idx),torch.nn.Dropout(self.dropout_rate))

        for idx, layer in enumerate(self.layers_neurons_[:-1]):
            if self.batch_norm:
                self.decoder.add_module("batch_norm"+str(idx),torch.nn.BatchNorm1d(self.layers_neurons_decoder_[idx]))
            self.decoder.add_module("linear"+str(idx),torch.nn.Linear(self.layers_neurons_decoder_[idx],
                                                                      self.layers_neurons_decoder_[idx+1]))
            self.encoder.add_module(self.hidden_activation+str(idx),self.activation)
            self.decoder.add_module("dropout"+str(idx),torch.nn.Dropout(self.dropout_rate))

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        # self.logistic = Net()

        self.classifier = nn.Sequential(
            # nn.Linear(self.L*self.K, 1),
            nn.Linear(1, 1),
            nn.Sigmoid()
        )

        self.mu1 = 0
        self.mu2 = 1
        self.sigma1 = 0.1
        self.sigma2 = 0.1

        self.device = (
            "cuda"
            if torch.cuda.is_available() and self.model_config['device'] == 'cuda'
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        self.A = torch.nn.Parameter(torch.tensor(torch.rand(1)), requires_grad=True)
        self.A.grad = torch.tensor(torch.rand(1))
        self.B = torch.nn.Parameter(torch.tensor(torch.rand(1)), requires_grad=True)

    def _logistic(self, loss):
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
    def a_to_w(self, A):
        rv1 = torch.distributions.normal.Normal(loc=torch.tensor(self.mu1), scale=torch.tensor(self.sigma1))
        rv2 = torch.distributions.normal.Normal(loc=torch.tensor(self.mu2), scale=torch.tensor(self.sigma2))
        ranks = torch.empty((A.size()[1]), dtype = torch.float)
        tmp = torch.argsort(A, dim=1, descending=False)
        ranks[tmp] = torch.arange(0,A.size()[1])/(A.size()[1]-1)
        w = torch.exp(rv1.log_prob(ranks))+torch.exp(rv2.log_prob(ranks))
        w = torch.div(w,torch.sum(w))
        w = w.to(self.device, non_blocking = True).float()
        w = w.unsqueeze(0)
        return w

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


        enc, dec = self.autoencoder_forward(x)

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
        bp = 1 - torch.prod(torch.pow(1-pij+1e-10, A).clip(min=0, max=1), dim=1)
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
            enc, dec = self.autoencoder_forward(data_inst_l1)
            l1 = torch.nn.PairwiseDistance(p=2)(data_inst_l1, dec)
            data_inst_l1 = data_inst_l1[torch.where(l1 < torch.quantile(l1, 1 - 0.0625, dim=0))]
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

#
# class milpuAtt_MNIST(nn.Module):
#
#     r"""
#     The milpuAttention model, attention mechanism was used for transforming instance probabilities into bag probability
#     """
#
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
#         super(milpuAtt_MNIST, self).__init__()
#
#         self.model_config = model_config
#
#         self.build_model()
#
#
#     def build_model(self):
#
#         r'''
#         Instance method for building milpuAttention model according to config
#
#         '''
#
#         self.L = self.model_config['attention']['L']
#         self.D = self.model_config['attention']['D']
#         self.K = self.model_config['attention']['K']
#
#         self.encoder = nn.Sequential(
#             # 28 x 28
#             nn.Conv2d(1, 4, kernel_size=5),
#             # 4 x 24 x 24
#             nn.ReLU(True),
#             nn.Conv2d(4, 8, kernel_size=5),
#             nn.ReLU(True),
#             # 8 x 20 x 20 = 3200
#             nn.Flatten(),
#             nn.Linear(3200, 10),
#             # 10
#             nn.Softmax(),
#         )
#
#         self.decoder = nn.Sequential(
#             # 10
#             nn.Linear(10, 2000),
#             # 400
#             nn.ReLU(True),
#             nn.Linear(2000, 4000),
#             # 4000
#             nn.ReLU(True),
#             nn.Unflatten(1, (10, 20, 20)),
#             # 10 x 20 x 20
#             nn.ConvTranspose2d(10, 10, kernel_size=5),
#             # 24 x 24
#             nn.ConvTranspose2d(10, 1, kernel_size=5),
#             # 28 x 28
#             nn.Sigmoid(),
#         )
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
#
#         self.logistic = Net()
#
#         self.classifier = nn.Sequential(
#             nn.Linear(self.L*self.K, 1),
#             nn.Sigmoid()
#         )
#
#     def autoencoder_forward(self, x):
#
#         r"""
#             Instance method to get modification probability on the site level from instance features.
#
#                 Args:
#                     x (torch.Tensor): A tensor representation of the instance features
#
#                 Returns:
#                     enc (torch.Tensor): A tensor representation of the autoencoder latent features
#                     dec (torch.Tensor): A tensor representation of the autoencoder reconstruct features
#         """
#
#         enc = self.encoder(x)
#         dec = self.decoder(enc)
#
#         return enc, dec
#
#     def _weightnoisyor(self,pij):
#         rv1 = torch.distributions.normal.Normal(loc=torch.tensor(self.mu1), scale=torch.tensor(self.sigma1))
#         rv2 = torch.distributions.normal.Normal(loc=torch.tensor(self.mu2), scale=torch.tensor(self.sigma2))
#         nbags = pij.size()[0]
#         ninstances = pij.size()[1]
#         pij = pij.reshape(nbags,ninstances)
#         ranks = torch.empty((nbags, ninstances), dtype = torch.float)
#         tmp = torch.argsort(pij, dim=1, descending=False)
#         for i in range(nbags):
#             ranks[i,tmp[i,:]] = torch.arange(0,ninstances)/(ninstances-1)
#         w = torch.exp(rv1.log_prob(ranks))+torch.exp(rv2.log_prob(ranks))
#         w = torch.div(w,torch.sum(w, dim = 1).reshape(nbags,1))
#         pij = pij.to(self.device, non_blocking = True).float()
#         w = w.to(self.device, non_blocking = True).float()
#         noisyor = 1 - torch.prod(torch.pow(1-pij+1e-10,w).clip(min = 0, max = 1), dim = 1)
#         return noisyor
#
#     def Attforward(self, x):
#
#         r'''
#         Instance method to get modification probability on the site level from instance features.
#
#                 Args:
#                         x (torch.Tensor): A tensor representation of the instance features
#                 Returns:
#                         bp (torch.Tensor): A tensor representation the bag probability
#                         pij (torch.Tensor): A tensor representation the instance probability
#                         A (torch.Tensor): A tensor containing attention weight of instance features
#
#         '''
#
#
#         enc, dec = self.autoencoder_forward(x)
#
#         l1 = torch.nn.PairwiseDistance(p=2)(torch.flatten(x, start_dim=1), torch.flatten(dec, start_dim=1))*0.1
#         # l1 = torch.stack([(1-ssim(x[idx].unsqueeze(0), dec[idx].unsqueeze(0))) for idx in range(len(x))])
#
#         pij = self.logistic(l1)
#         #
#         # H = self.feature_extractor_part1(x)
#         # H = H.view(-1, 50 * 4 * 4)
#         # H = self.feature_extractor_part2(H)  # NxL
#
#         A = self.attention(enc)  # NxK
#         A = torch.transpose(A, 1, 0)  # KxN
#         A = F.softmax(A, dim=1)  # softmax over N
#
#         M = torch.mm(A, enc)
#         bc = self.classifier(M)
#
#         bp = 1 - torch.prod(torch.pow(1-pij+1e-10,A).clip(min=0, max=1), dim=1)
#         # bp = (1 - torch.prod((1-pij+1e-10).clip(min=0, max=1))).unsqueeze(0)
#
#         return bc, bp, pij, A
#
#     def _log_diverse_density(self, pi, y_bags):
#         r'''
#         Instance method to Compute the likelihood given bag labels y_bags and bag probabilities pi.
#                 Args:
#                         pi (torch.Tensor): A tensor representation of the bag probabilities
#                         y_bags (torch.Tensor): A tensor representation of the bag labels
#                 Returns:
#                         likelihood (torch.Tensor): A tensor representation of the likelihood
#
#         '''
#
#         z = torch.where(y_bags == -1)[0]
#         if z.nelement() > 0:
#             zero_sum = torch.sum(torch.log(1 - pi[z] + 1e-10))
#         else:
#             zero_sum = torch.tensor(0).float()
#
#         o = torch.where(y_bags == 1)[0]
#         if o.nelement() > 0:
#             one_sum = torch.sum(torch.log(pi[o] + 1e-10))
#         else:
#             one_sum = torch.tensor(0).float()
#         return zero_sum + one_sum
#
#     def bag_forward(self, input):
#
#         r'''
#         Instance method to get modification probability on the bag level from instance features.
#
#                Args:
#                        input (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
#                        Tensor representation of bag, bag_labels, instance labels and number of instance
#                Returns:
#                        loss (torch.Tensor): A tensor representation of model loss
#
#        '''
#
#
#         bag, bag_labels = input
#
#         idx_l1 = torch.where(bag_labels != 1)[0]
#
#         if idx_l1.shape[0] > 0:
#             data_inst_l1 = torch.concat([bag[index] for index in idx_l1])
#             enc, dec = self.autoencoder_forward(data_inst_l1)
#             loss1 = torch.nn.MSELoss()(data_inst_l1, dec)
#         else:
#             loss1 = torch.tensor(0, dtype=torch.float)  # reconstruct loss
#
#         data_inst = [self.Attforward(item) for item in bag]
#         idx_l2 = torch.where(bag_labels != 0)[0]
#         if idx_l2.shape[0] > 0:
#             l2 = [data_inst[item] for item in idx_l2]
#             pc = torch.concat([item[0] for item in l2]).squeeze()
#             pi = torch.stack([item[1] for item in l2])
#             y = torch.stack([bag_labels[index] for index in idx_l2])
#             yi = torch.clone(y)
#             yi[torch.where(yi == -1)] = 0
#             loss2 = -1*(self._log_diverse_density(pi, y)+1e-10) + 0.01*(self.logistic.A**2+self.logistic.B**2)[0]
#             loss3 = torch.sum(-1. * (yi * torch.log(pc) + (1. - yi) * torch.log(1. - pc)))
#
#         else:
#             loss2 = 0.01*(self.logistic.A**2+self.logistic.B**2)[0]
#             loss3 = torch.tensor(0, dtype=torch.float)
#
#         # bag_pi = torch.stack([item[0] for item in data_inst])
#
#         return loss1, loss2, loss3, data_inst
#


