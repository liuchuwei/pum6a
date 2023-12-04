import torch.nn as nn
from typing import *
import torch
import torch.nn.functional as F

class iAE(nn.Module):

    r"""
    The iAE model.

    """
    def __init__(self, model_config: Dict):

        r"""
        Initialization function for the class

            Args:
                    model_config (Dict): A dictionary containing model configurations.

            Returns:
                    None
        """

        super(iAE, self).__init__()

        self.device = (
            "cuda"
            if torch.cuda.is_available() and model_config['device'] == 'cuda'
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        self.model_config = model_config
        self.pesu_bag = None
        self.build_model()

    def get_activation_by_name(self, name):

        r'''
        Instance method for building activation function

            Args:
                name (str): activation name. possible value: relu、sigmoid、 tanh

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
        self.encoder_activation = self.get_activation_by_name(
            self.model_config['autoencoder']['encoder_activation'])
        self.decoder_activation = self.get_activation_by_name(
            self.model_config['autoencoder']['decoder_activation'])

        hidden_neurons = self.model_config['autoencoder']['hidden_neurons']
        self.layers_neurons_ = [1, *hidden_neurons]
        self.layers_neurons_decoder_ = self.layers_neurons_[::-1]
        self.latent = self.model_config['autoencoder']['latent']

        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()

        for idx, layer in enumerate(self.layers_neurons_[:-1]):
            self.encoder.add_module("conv" + str(idx),
                                    nn.Conv2d(self.layers_neurons_[idx], self.layers_neurons_[idx + 1],
                                              kernel_size=self.kernal_size))
            self.encoder.add_module("hidden_activation" + str(idx), self.hidden_activation)

        self.encoder.add_module("faltten", nn.Flatten())
        n_out_feat = (self.n_features[0] - len(hidden_neurons) * (self.kernal_size - 1))
        n_linear = n_out_feat ** 2 * hidden_neurons[-1]
        self.encoder.add_module("linear_last", nn.Linear(n_linear, self.latent))
        self.encoder.add_module("out_acitvation", self.encoder_activation)

        self.decoder.add_module("linear", nn.Linear(self.latent, n_linear))
        self.decoder.add_module("input_activation", self.hidden_activation)
        self.decoder.add_module("unflatten",
                                nn.Unflatten(1, (self.layers_neurons_decoder_[0], n_out_feat, n_out_feat)))

        for idx, layer in enumerate(self.layers_neurons_decoder_[:-1]):
            self.decoder.add_module("convT" + str(idx),
                                    nn.ConvTranspose2d(self.layers_neurons_decoder_[idx],
                                                       self.layers_neurons_decoder_[idx + 1],
                                                       kernel_size=self.kernal_size))
            self.decoder.add_module("hidden_activation" + str(idx), self.hidden_activation)

        self.encoder.add_module("out_acitvation", self.decoder_activation)

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

        '''

        Instance method for building logistic module

        '''

        self.A = torch.nn.Parameter(torch.tensor(torch.rand(1)), requires_grad=True)
        self.A.grad = torch.tensor(torch.rand(1))
        self.B = torch.nn.Parameter(torch.tensor(torch.rand(1)), requires_grad=True)


    def _logistic(self, att):

        """
        instance method to calculate instance probability according attention weight

            Args:
                att (torch.Tensor): Tensor representation of attention weight
        """

        return torch.sigmoid(self.A * att + self.B)

    def build_model(self):

        r'''
        Instance method for building pum6a model according to config
        '''

        self.confactor = self.model_config['confactor']

        '1. build autoencoder'
        self.build_AE()

        '2. build logistic module'
        self.build_logistic()

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

        Y_prob = self._weightnoisyor(pij)

        return Y_prob, pij

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
            data_inst_l1 = torch.concat([bag[item] for item in idx_l1])
            enc = self.encoder(data_inst_l1)
            dec = self.decoder(enc)
            l1 = torch.nn.PairwiseDistance(p=2)(data_inst_l1, dec)
            data_inst_l1 = data_inst_l1[torch.where(l1 < torch.quantile(l1, 1 - self.confactor, dim=0))]
            enc = self.encoder(data_inst_l1)
            dec = self.decoder(enc)
            loss = torch.nn.MSELoss()(data_inst_l1, dec)
        else:
            loss = 0.01 * (self.A ** 2 + self.B ** 2)[0]

        data_inst = [self._forward(item) for item in bag]
        idx_l2 = torch.where(bag_labels == -1)[0]
        idx_l3 = torch.where(bag_labels == 1)[0]
        if idx_l2.shape[0] > 0:
            l2 = torch.concat([bag[item] for item in idx_l1])
            enc = self.encoder(l2)
            dec = self.decoder(enc)
            l2_dist = torch.nn.PairwiseDistance(p=2)(l2, dec)
            loss += l2_dist.mean()

            if idx_l3.shape[0]>0:

                l3_dist = []
                for item in idx_l3:
                    enc = self.encoder(bag[item])
                    dec = self.decoder(enc)
                    l3_dist.append(torch.nn.PairwiseDistance(p=2)(bag[item], dec).max())

                iAUC_loss = 0
                for i in l3_dist:

                    for j in l2_dist:

                        iAUC_loss += torch.nn.Sigmoid()(i - j)

                iAUC_loss /= (len(l2_dist)*len(l3_dist))
                loss += iAUC_loss

        return loss, data_inst


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
        self.pesu_bag = None
        self.build_model()

    def get_activation_by_name(self, name):

        r'''
        Instance method for building activation function

            Args:
                name (str): activation name. possible value: relu、sigmoid、 tanh

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

            raise ValueError('invalid autoencoder type!')


    def build_attention(self):

        r'''
        Instance method for building attention module according to config
        '''

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

    def build_classifier(self):

        r'''
        Instance method for building classifier module according to config
        '''


        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, 1),
            nn.Sigmoid()
        )

    def build_logistic(self):

        '''

        Instance method for building logistic module

        '''

        self.A = torch.nn.Parameter(torch.tensor(torch.rand(1)), requires_grad=True)
        self.A.grad = torch.tensor(torch.rand(1))
        self.B = torch.nn.Parameter(torch.tensor(torch.rand(1)), requires_grad=True)


    def _logistic(self, att):

        """
        instance method to calculate instance probability according attention weight

            Args:
                att (torch.Tensor): Tensor representation of attention weight
        """

        return torch.sigmoid(self.A * att + self.B)


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

        # '4. build autoencoder module'
        # self.build_AE()

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
        nbags = pij.size()[0]
        ninstances = pij.size()[1]
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
                        Y_prob (torch.Tensor): A tensor representation the bag probability
                        A (torch.Tensor): A tensor containing attention weight of instance features

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

        pi = self._weightnoisyor(pij)
        # pi = 1 - torch.prod(1-pij, dim = 1)

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
                       loss (torch.Tensor): A tensor representation of model loss

       '''


        bag, bag_labels, n_instance = input

        # idx_l1 = torch.where(bag_labels != 1)[0]
        # if idx_l1.shape[0] > 0:
        #     data_inst = torch.concat([bag[item] for item in idx_l1])
        #     enc = self.encoder(data_inst)
        #     dec = self.decoder(enc)
        #     l1 = torch.nn.PairwiseDistance(p=2)(data_inst, dec)
        #     data_inst_l1 = data_inst[torch.where(l1 < torch.quantile(l1, 1 - 0.06, dim=0))]
        #     enc = self.encoder(data_inst_l1)
        #     dec = self.decoder(enc)
        #     loss1 = torch.nn.MSELoss()(data_inst_l1, dec)
        # else:
        #     loss1 = 0.01*(self.A**2+self.B**2)[0]
        #
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
            # loss += -1*(self._log_diverse_density(p, y)+1e-10)
        else:
            loss = 0.01*(self.A**2+self.B**2)[0]

        return loss, data_inst



class puma(nn.Module):

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
                                    nn.Conv2d(self.layers_neurons_[idx], self.layers_neurons_[idx + 1],
                                              kernel_size=self.kernal_size))
            self.encoder.add_module("hidden_activation" + str(idx), self.hidden_activation)

        self.encoder.add_module("faltten", nn.Flatten())
        n_out_feat = (self.n_features[0] - len(hidden_neurons) * (self.kernal_size - 1))
        n_linear = n_out_feat ** 2 * hidden_neurons[-1]
        self.encoder.add_module("linear_last", nn.Linear(n_linear, self.latent))
        self.encoder.add_module("out_acitvation", self.encoder_activation)

        self.decoder.add_module("linear", nn.Linear(self.latent, n_linear))
        self.decoder.add_module("input_activation", self.hidden_activation)
        self.decoder.add_module("unflatten", nn.Unflatten(1, (self.layers_neurons_decoder_[0], n_out_feat, n_out_feat)))

        for idx, layer in enumerate(self.layers_neurons_decoder_[:-1]):
            self.decoder.add_module("convT" + str(idx),
                                    nn.ConvTranspose2d(self.layers_neurons_decoder_[idx],
                                                       self.layers_neurons_decoder_[idx + 1],
                                                       kernel_size=self.kernal_size))
            self.decoder.add_module("hidden_activation" + str(idx), self.hidden_activation)

        self.encoder.add_module("out_acitvation", self.decoder_activation)

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

        self.confactor = self.model_config['confactor']

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

        Y_prob = self._weightnoisyor(pij)

        return Y_prob, pij

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
            data_inst_l1 = torch.concat([bag[item] for item in idx_l1])
            enc = self.encoder(data_inst_l1)
            dec = self.decoder(enc)
            l1 = torch.nn.PairwiseDistance(p=2)(data_inst_l1, dec)
            data_inst_l1 = data_inst_l1[torch.where(l1 < torch.quantile(l1, 1 - self.confactor), dim=0)]
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
            loss += -1 * (self._log_diverse_density(pi, y) + 1e-10)
        else:
            loss += 0.01*(self.A**2+self.B**2)[0]

        return loss, data_inst

