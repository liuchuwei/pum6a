from typing import *
import numpy as np


def genData(k, nbags, bag_contfactor=0.1, seed=331):

    """
    Method for generating bag dataset
    """
    np.random.seed(seed)
    norm_clusters = {1: ([1, 1], [2, .2]), 2: ([-1, 8], [1.25, 1.75]), 3: ([10, -1], [0.5, 2]), 4: ([5, -2], [1, 2]),
                     5: ([3, 2], [0.5, 2]), 6: ([-8, 0], [0.75, 1]), 7: ([-10, -5], [1, 1]), 8: ([0, 0], [0.5, 0.5]),
                     9: ([5, -5], [1.5, 0.5]), 10: ([-5, 5], [0.5, 1])}
    anom_clusters = {11: ([7, 10], [2, 2]), 12: ([-3, -4], [1.5, 1.5]), 13: ([-10, 12], [1.5, 1.5]),
                     14: ([-14, 4], [.25, 3])}

    n_norm_clusters = len(norm_clusters.keys())
    n_anom_clusters = len(anom_clusters.keys())
    tot_clusters = n_norm_clusters + n_anom_clusters

    bags_labels = np.zeros(nbags, int)
    bags = {}
    X_inst = np.empty(shape=(0, 2))
    y_inst = np.array([])

    for b in range(nbags):
        label = np.random.binomial(1, bag_contfactor, size=1)[0]
        w = np.zeros(tot_clusters, float)
        if label == 0:
            w = np.zeros(n_norm_clusters, float)
            tmp_norm_cls = int(
                np.round(np.random.uniform(low=0.5 / n_norm_clusters, high=1.0, size=1)[0] * n_norm_clusters, 0))
            chosen_normcls = np.random.choice(np.arange(0, n_norm_clusters), tmp_norm_cls, replace=False)
            w[chosen_normcls] = np.random.uniform(low=0.0, high=1.0, size=tmp_norm_cls)
            w = w / sum(w)
            w = np.around(w * k).astype(int)
            w[np.random.choice(np.where(w >= sum(w) - k)[0], size=1, replace=False)] += k - sum(w)
            X, y = gen_normals(norm_clusters, w)
        elif label == 1:
            while sum(w[-n_anom_clusters:]) == 0:
                w = np.zeros(tot_clusters, float)
                tmp_norm_cls = int(
                    np.round(np.random.uniform(low=0.5 / n_norm_clusters, high=1.0, size=1)[0] * n_norm_clusters, 0))
                chosen_normcls = np.random.choice(np.arange(0, n_norm_clusters), tmp_norm_cls, replace=False)
                tmp_anom_cls = int(
                    np.round(np.random.uniform(low=0.5 / n_anom_clusters, high=1.0, size=1)[0] * n_anom_clusters, 0))
                chosen_anomcls = np.random.choice(np.arange(n_norm_clusters, tot_clusters), tmp_anom_cls, replace=False)
                idx_w = np.concatenate((chosen_normcls, chosen_anomcls))
                w[idx_w] = np.random.uniform(low=0.0, high=1.0, size=tmp_norm_cls + tmp_anom_cls)
                w = w / sum(w)
                w = np.around(w * k).astype(int)
                w[np.random.choice(np.where(w >= sum(w) - k)[0], 1)] += k - sum(w)
            X, y = gen_anomalies(norm_clusters, anom_clusters, w)
            bags_labels[b] = 1
        bags[b] = X.T
        X_inst = np.concatenate((X_inst, X.T))
        y_inst = np.concatenate((y_inst, y))
    return bags, bags_labels, X_inst, y_inst


def gen_normals(norm_clusters, w):

    """
    Method for generating normal bag
    """
    X1 = np.array([])
    X2 = np.array([])

    for key, val in norm_clusters.items():
        X1_mean = val[0][0]
        X1_var = val[1][0]
        X2_mean = val[0][1]
        X2_var = val[1][1]

        X1 = np.concatenate((X1, np.random.normal(loc=X1_mean, scale=X1_var, size=w[key - 1])))
        X2 = np.concatenate((X2, np.random.normal(loc=X2_mean, scale=X2_var, size=w[key - 1])))

    X = np.array([X1, X2]).reshape(2, -1)
    y = np.zeros(sum(w), int)
    return X, y


def gen_anomalies(norm_clusters, anom_clusters, w):

    """
    Method for generating abnormal bag

    """
    bag_clusters = {**norm_clusters, **anom_clusters}

    X1 = np.array([])
    X2 = np.array([])

    for key, val in bag_clusters.items():
        X1_mean = val[0][0]
        X1_var = val[1][0]
        X2_mean = val[0][1]
        X2_var = val[1][1]

        X1 = np.concatenate((X1, np.random.normal(loc=X1_mean, scale=X1_var, size=w[key - 1])))
        X2 = np.concatenate((X2, np.random.normal(loc=X2_mean, scale=X2_var, size=w[key - 1])))
    nnormals = sum(w[:-len(anom_clusters.keys())])
    nanom = sum(w[-len(anom_clusters.keys()):])
    y = np.zeros(nnormals + nanom, int)
    y[-nanom:] = 1
    X = np.array([X1, X2]).reshape(2, -1)
    return X, y
