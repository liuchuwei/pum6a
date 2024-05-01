from typing import *
import torch
import torch.nn as nn
import time
import itertools

from sklearn import svm

from model_factory import MI
import numpy as np
from scipy import stats

def GetActivation(name):
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



def build_attention(L: Optional[int]=500,
                    D: Optional[int]=128,
                    K: Optional[int]=1):

    r'''
    Instance method for building attention module according

        Args:
            L (int):  attention module input nodes
            D (int):  attention module intermediate nodes
            K (int):  attention module output nodes

        Return:
            attention (torch.nn.Sequential): attention module
    '''

    attention = nn.Sequential(
        nn.Linear(L, D),
        nn.Tanh(),
        nn.Linear(D, K)
    )

    return attention

def build_classifier(input: Optional[int]=500):

    r'''
    Instance method for building classifier module according

        Args:
            input (int):  attention module input nodes

        Return:
            classifier (torch.nn.Sequential): classifier module
    '''


    classifier = nn.Sequential(
        nn.Linear(input, 1),
        nn.Sigmoid()
    )

    return classifier


def build_logistic(device: Optional[str]='cuda'):

    '''

    Instance method for building logistic module

        Args:
            device (str): whether to use cuda for calculation

        Return:
            A (torch.nn.Parameter): weight parameter of logistic module
            B (torch.nn.Parameter): bias parameter of logistic module

    '''

    A = torch.nn.Parameter(torch.tensor(torch.rand(1, device=device)), requires_grad=True)
    A.grad = torch.tensor(torch.rand(1, device=device))
    B = torch.nn.Parameter(torch.tensor(torch.rand(1, device=device)), requires_grad=True)

    return A, B

def weightnoisyor(pij: Optional[list] = None,
                  device: Optional[str] = 'cuda',
                  mu1: Optional[int] = 0,
                  mu2: Optional[int] = 1,
                  sigma1: Optional[int] = 0.1,
                  sigma2: Optional[int] = 0.1,
                  ):

    """
    instance method to calculate bag probability based on instance probability

        Args:
            pij (torch.Tensor): Tensor representation of instance probability
            device (str): Whether to use gpu for calculation
            mu1 (int): loc of torch normal distributions
            mu2 (int): loc of torch normal distributions
            sigma1 (int): scale of torch normal distributions
            sigma1 (int): scale of torch normal distributions
        Return:
            noisyor (torch.Tensor): Tensor representation of bag probability
    """

    rv1 = torch.distributions.normal.Normal(loc=torch.tensor(mu1), scale=torch.tensor(sigma1))
    rv2 = torch.distributions.normal.Normal(loc=torch.tensor(mu2), scale=torch.tensor(sigma2))
    nbags = 1
    ninstances = pij.size()[1]
    pij = pij.reshape(nbags, ninstances)
    ranks = torch.empty((nbags, ninstances), dtype=torch.float)
    tmp = torch.argsort(pij, dim=1, descending=False)
    for i in range(nbags):
        ranks[i,tmp[i,:]] = torch.arange(0,ninstances)/(ninstances-1)
    w = torch.exp(rv1.log_prob(ranks))+torch.exp(rv2.log_prob(ranks))
    w = torch.div(w,torch.sum(w, dim=1).reshape(nbags,1))
    pij = pij.to(device, non_blocking = True).float()
    w = w.to(device, non_blocking = True).float()
    noisyor = 1 - torch.prod(torch.pow(1-pij+1e-10,w).clip(min=0, max=1), dim=1)

    return noisyor


def log_diverse_density(pi, y_bags):
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

class FeatureExtractor(object):

    r"""
    Object class for building linear or conv feature extractor
    """

    def __init__(self, model_config: Dict):

        r"""
        Initialization function for the class

            Args:
                    model_config (Dict): A dictionary containing model_factory configurations.

            Returns:
                    None
        """

        self.model_config = model_config
        self.build_FE()

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
            self.feature_extractor.add_module(self.hidden_activation + str(idx), GetActivation(self.hidden_activation))
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
            self.feature_extractor_1.add_module("hidden_activation" + str(idx), GetActivation(self.hidden_activation))
            self.feature_extractor_1.add_module("pool" + str(idx),
                                    nn.MaxPool2d(self.pool_kernal_size, stride=self.pool_stride))

        self.L = self.model_config['attention']['L']
        self.linear_input = self.model_config['feature_extractor']['linear_input']
        self.feature_extractor_2.add_module("linear",
                                            torch.nn.Linear(self.linear_input, self.L))
        self.feature_extractor_2.add_module("hidden_activation", GetActivation(self.hidden_activation))

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

            raise ValueError('invalid feature extractor type!')


class AutoEncoder(object):

    r"""
    Object class for building AutoEncoder
    """

    def __init__(self, model_config: Dict):

        r"""
        Initialization function for the class

            Args:
                    model_config (Dict): A dictionary containing model_factory configurations.

            Returns:
                    None
        """

        self.model_config = model_config
        self.build_AE()

    def build_linearAE(self):

        r'''
        Instance method for building linear autoencoder module according to config
        '''

        self.n_features = self.model_config['autoencoder']['n_features']
        self.dropout_rate = self.model_config['autoencoder']['dropout_rate']
        self.batch_norm = self.model_config['autoencoder']['batch_norm']
        self.hidden_activation = self.model_config['autoencoder']['hidden_activation']

        self.activation = self.hidden_activation
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
            self.encoder.add_module(self.hidden_activation + str(idx), GetActivation(self.activation))
            self.encoder.add_module("dropout" + str(idx), torch.nn.Dropout(self.dropout_rate))

        for idx, layer in enumerate(self.layers_neurons_[:-1]):
            if self.batch_norm:
                self.decoder.add_module("batch_norm" + str(idx),
                                        torch.nn.BatchNorm1d(self.layers_neurons_decoder_[idx]))
            self.decoder.add_module("linear" + str(idx), torch.nn.Linear(self.layers_neurons_decoder_[idx],
                                                                         self.layers_neurons_decoder_[idx + 1]))
            self.encoder.add_module(self.hidden_activation + str(idx), GetActivation(self.activation))
            self.decoder.add_module("dropout" + str(idx), torch.nn.Dropout(self.dropout_rate))

    def build_convAE(self):

        r'''
        Instance method for building linear autoencoder module according to config
        '''

        self.n_features = self.model_config['autoencoder']['n_features']
        self.kernal_size = self.model_config['autoencoder']['kernal_size']
        self.hidden_activation = self.model_config['autoencoder']['hidden_activation']
        self.encoder_activation = self.model_config['autoencoder']['encoder_activation']
        self.decoder_activation = self.model_config['autoencoder']['decoder_activation']

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
            self.encoder.add_module("hidden_activation" + str(idx), GetActivation(self.hidden_activation))

        self.encoder.add_module("faltten", nn.Flatten())
        n_out_feat = (self.n_features[0] - len(hidden_neurons) * (self.kernal_size - 1))
        n_linear = n_out_feat ** 2 * hidden_neurons[-1]
        self.encoder.add_module("linear_last", nn.Linear(n_linear, self.latent))
        self.encoder.add_module("out_acitvation", GetActivation(self.encoder_activation))

        self.decoder.add_module("linear", nn.Linear(self.latent, n_linear))
        self.decoder.add_module("input_activation", GetActivation(self.hidden_activation))
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


def train_lsdd(data, args):
  widths = [1.0e-2, 1.0e-4, 1.0e-6]
  regs = [1.0, 1.0e-03, 1.0e-06]

  def train(data, width, reg, measure_time = False):
    if measure_time:
      t_start = time.time()

    model = model_factory.MI.UU.LSDD.train(data, width, reg, args)
    metadata = {'width': width, 'reg': reg}

    if measure_time:
      t_end = time.time()
      print("#  elapsed time = {}".format(t_end - t_start))

    return model, metadata

  # cross validation
  best_param = {}
  best_error = np.inf
  if args['verbose']:
    print("# *** Cross Validation ***")
  for width, reg in itertools.product(widths, regs):
    errors = []
    for data_train, data_val in MI.cross_validation(data, 5):
      t = model_factory.MI.UU.LSDD.LSDD(
          np.vstack(MI.extract_bags(data_train, 1)),
          np.vstack(MI.extract_bags(data_train, 0)),
          width, reg)
      e = model_factory.MI.UU.LSDD.validation_error(data_val, data_train, width, reg, t)
      errors.append(e)

    error = np.mean(errors)

    if args['verbose']:
      print("#  width = {:.3e} / reg = {:.3e} / error = {:.3e}".format(width, reg, error))

    if error < best_error:
      best_error = error
      best_param = {'width': width, 'reg': reg}

  if args['verbose']:
    print("# {}".format('-'*80))

  model, metadata = train(data, best_param['width'], best_param['reg'], measure_time=True)

  return model, best_param


def train_pu_skc(data, args):
  degs = [1, 2, 3]
  regs = [1.0, 1.0e-03, 1.0e-06]

  def train(data, deg, reg, measure_time = False):
    if measure_time:
      t_start = time.time()

    bdim = len(data)
    theta = MI.PU.class_prior(data, degree = 1, reg = 1.0e+05)
    basis = model_factory.MI.kernel.minimax_basis(data, deg)
    model = model_factory.MI.PU.SKC.train(data, basis, bdim, theta, reg, args)
    metadata = {'theta': theta, 'reg': reg, 'degree': deg}

    if measure_time:
      t_end = time.time()
      print("#  elapsed time = {}".format(t_end - t_start))

    return model, metadata

  # cross validation
  best_param = {'degree': None, 'reg': None}
  best_error = np.inf
  if args['verbose']:
    print("# *** Cross Validation ***")
  for deg, reg in itertools.product(degs, regs):
    try:
      errors = []
      for data_train, data_val in MI.cross_validation(data, 5):
        clf, metadata = train(data_train, deg, reg)
        e = MI.PU.prediction_error(data_val, clf, metadata['theta'])
        errors.append(e)

      error = np.mean(errors)

      if args['verbose']:
        print("#  degree = {} / reg = {:.3e} : theta = {:.3e} / error = {:.3e}".format(deg, reg, metadata['theta'], error))

      if error < best_error:
        best_error = error
        best_param = {'degree': deg, 'reg': reg}

    except ValueError:
      # sometimes fails for large degree
      if args['verbose']:
        print("#  degree = {} / reg = {:.3e} : error = NaN".format(deg, reg))

  if args['verbose']:
    print("# {}".format('-'*80))

  # training using the best parameter
  model, metadata = train(data, best_param['degree'], best_param['reg'], measure_time = True)

  if args['verbose']:
    print("#  estimated class prior = {:.6f}".format(metadata['theta']))

  return model, best_param


def train_dsdd(data, args):
  widths = [1.0e-2, 1.0e-4, 1.0e-6]
  regs = [1.0, 1.0e-03, 1.0e-06]

  def train(data, width, reg, measure_time = False):
    if measure_time:
      t_start = time.time()

    model = model_factory.MI.UU.DSDD.train(data, width, reg, args)
    metadata = {'width': width, 'reg': reg}

    if measure_time:
      t_end = time.time()
      print("#  elapsed time = {}".format(t_end - t_start))

    return model, metadata


  model, metadata = train(data, 1.0e-2, 1.0e-2, measure_time=True)

  return model


pca_dim = 30


def reliable_negative_bag_idx(bags, uidx, w, N):
  sorted_confs = sorted(list(zip(uidx, w[uidx])), key=lambda x:x[1], reverse=True)
  return [i for i, _ in sorted_confs[:N]]


def weighted_kde(Bn, conf):
  # N.B. apply WKDE function to the instance multiplied by its confindence value
  weighted_ins = np.vstack([conf[i] * B['instance'][0] for i, B in enumerate(Bn)])
  return stats.gaussian_kde(weighted_ins.T)


def form_pmp(bags, conf, pidx, nidx, Dn):
  choose_witness = lambda bags, conf: np.array(torch.stack([
    # for each bag
    bags[i]['instance'][
      # choose the least negative instance
      min(
        [(j, Dn(conf[i] * np.array(x))) for j, x in enumerate(bags[i]['instance'].tolist())],
        key=lambda pair: pair[1]
      )[0]
    ]
    for i in range(len(bags))
  ]))

  p_bags = [bags[i] for i in pidx]
  n_bags = [bags[i] for i in nidx]
  p_conf = [conf[i] for i in pidx]
  n_conf = [conf[i] for i in nidx]

  X = np.r_[choose_witness(p_bags, p_conf), choose_witness(n_bags, n_conf)]
  Y = np.r_[np.ones(len(p_bags)), -1 * np.ones(len(n_bags))]
  W = np.r_[p_conf, n_conf]

  return X, Y, W


def pumil_clf_wrapper(clf, n_dist, learning_phase = False):
  """
  Parameters
  ----------
  clf : instance classifier
  n_dist : (estimated) distribution of negative instances
  """
  witness = lambda xs, w: xs[
    min(
      [(j, n_dist(w * np.array(x))) for j, x in enumerate(xs.tolist())],
      key = lambda pair: pair[1]
    )[0]
  ].reshape(1, -1)

  if learning_phase:
    return lambda bag, conf: clf(witness(bag['instance'], conf))

  else:
    # N.B. fix the confidence of test bag to 1
    return lambda instances: clf(witness(instances, 1))


def affinity(clf, conf, bags, uidx, nidx):
  # evaluate F-score on unlabeled set
  # regard "reliable negative bags" as negative set, and the other bags as positive set
  pidx = list(set(uidx) - set(nidx))
  pred = np.array([clf(bags[i], conf[i]) for i in pidx + nidx])
  true = np.r_[np.ones(len(pidx)), -1 * np.ones(len(nidx))]

  return MI.f_score(pred, true)


def train_pumil_clf(bags, pidx, uidx, w, NL, learning_phase = False):
  # top-{NL} reliable negative bags
  relnidx = reliable_negative_bag_idx(bags, uidx, w, NL)
  Bn = [bags[j] for j in relnidx]
  # estimated p(X|Y=-1) via WKDE
  Dn = weighted_kde(Bn, w[relnidx])
  # form Positive Margin Pool (PMP)
  pmp_x, pmp_y, pmp_conf = form_pmp(bags, w, pidx, relnidx, Dn)
  # train SVM by using PMP instances
  pmp_weighted_x = np.multiply(pmp_x.T, pmp_conf).T
  clf = svm.LinearSVC(loss = 'hinge')
  clf.fit(pmp_weighted_x, pmp_y)
  clf_ = pumil_clf_wrapper(lambda x: float(clf.decision_function(x)), Dn, learning_phase)

  if learning_phase:
    return clf_, relnidx

  else:
    return clf_


def pumil(bags, NL, NU, args):
  L = 2      # the number of confidence vectors
  # L = 100      # the number of confidence vectors
  c = 0.1      # clone factor
  T = 1.0e-08  # threshold (eps)
  M = 1e+10    # max iteration limit

  # list of indices
  pidx = [i for i, B in enumerate(bags) if B['label'][0] == 1]
  uidx = [i for i, B in enumerate(bags) if B['label'][0] == 0]

  # initialization
  W = np.ones((L, NL + NU))
  for i in range(L):
    for j in uidx:
      # assign random confidences to unlabeled bags
      W[i][j] = np.random.uniform(0, 1)

  # training phase
  f = np.zeros(L)   # affinity scores for each confidence vector
  t = 1             # current epoch
  delta = 0         # the difference between best affinities of last and current epoch
  best_score = 0    # the best affinity score of the last epoch
  best_conf = None  # the best confidence weight vector

  while (t == 1) or (t <= M and delta >= T):
    for i in range(L):
      # obtain classifier from confidence vector
      clf, relnidx = train_pumil_clf(bags, pidx, uidx, W[i], NL, learning_phase = True)
      # calculate affinity scores
      f[i] = affinity(clf, W[i], bags, uidx, relnidx)

    best = f.argmax()

    # antibody clone
    for i in range(L):
      if i != best and stats.bernoulli(c) == 1:
        W[i] = W[best]

    # antibody mutation
    V = W + np.multiply(
      (W - np.tile(W[best], (L, 1))).T,
      (np.ones(L) - f) * np.random.uniform(0, 1, size=L)
    ).T

    # antibody update (whether to accept mutation or not)
    for i in range(L):
      # evaluate proposed confidence vector
      clf, relnidx = train_pumil_clf(bags, pidx, uidx, V[i], NL, learning_phase = True)
      f_ = affinity(clf, V[i], bags, uidx, relnidx)
      if f_ > f[i]:
        W[i] = V[i]
        f[i] = f_

    delta = f[best] - best_score
    best_score = f[best]
    best_conf  = W[best]

    t += 1

  return train_pumil_clf(bags, pidx, uidx, best_conf, NL)
