from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
import numpy as np
from neural_nets.input_to_onehot import input_to_onehot
from neural_nets.models.cross_entropy import input_cross_entropy


def cluster_bayesian_gmm(onehot_input):
    bgmm = BayesianGaussianMixture(n_components=10).fit(onehot_input)
    input = bgmm.predict_proba(onehot_input)
    input = np.array(input)

    return input


def cluster_gmm(onehot_input):
    gmm = GaussianMixture(n_components=30).fit(onehot_input)
    input = gmm.predict_proba(onehot_input)
    input = np.array(input)

    return input