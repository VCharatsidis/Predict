import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from input_cross_entropy import cross_entropy_input_to_onehot

# dimensionality of hidden layer
h = 50
# K mixing components (PRML p. 274)
# Can also formulate as a K-dimensional, one-hot
# encoded, latent variable $$z$$, and have the model
# produce values for $$\mu_k = p(z_k = 1)$$, i.e., the
# prob of each possible state of $$z$$. (PRML p. 430)
k = 30  # 3
# We specialize to the case of isotropic covariances (PRML p. 273),
# so the covariance matrix is diagonal with equal diagonal elements,
# i.e., the variances for each dimension of y are equivalent.
# therefore, the MDN outputs pi & sigma scalars for each mixture
# component, and a mu vector for each mixture component containing
# means for each target variable.
# NOTE: we could use the shorthand `d_out = 3*k`, since our target
# variable for this project only has a dimensionality of 1, but
# the following is more general.
# d_out = (t + 2) * k  # t is L from PRML p. 274
# NOTE: actually cleaner to just separate pi, sigma^2, & mu into
# separate functions.
n = 803
d = 25
t = 1
d_pi = k
d_sigmasq = k
d_mu = t * k

# w1 = Variable(torch.randn(d, h) * np.sqrt(1/d), requires_grad=True)
# #w1 = Variable((torch.randn(d, h) * np.sqrt(2/(d+h))).double().requires_grad_())
# b1 = Variable(torch.zeros(1, h).double().requires_grad_())
# w_pi = Variable((torch.randn(h, d_pi) * np.sqrt(2/(d+h))).double().requires_grad_())
# b_pi = Variable(torch.zeros(1, d_pi).double().requires_grad_())
# w_sigmasq = Variable((torch.randn(h, d_sigmasq) * np.sqrt(2/(d+h))).double().requires_grad_())
# b_sigmasq = Variable(torch.zeros(1, d_sigmasq).double().requires_grad_())
# w_mu = Variable((torch.randn(h, d_mu) * np.sqrt(2/(d+h))).double().requires_grad_())
# b_mu = Variable(torch.zeros(1, d_mu).double().requires_grad_())

w1 = Variable(torch.randn(d, h) * np.sqrt(2/(d+h)), requires_grad=True)
b1 = Variable(torch.zeros(1, h), requires_grad=True)
w_pi = Variable(torch.randn(h, d_pi) * np.sqrt(2/(d+h)), requires_grad=True)
b_pi = Variable(torch.zeros(1, d_pi), requires_grad=True)
w_sigmasq = Variable(torch.randn(h, d_sigmasq) * np.sqrt(2/(d+h)), requires_grad=True)
b_sigmasq = Variable(torch.zeros(1, d_sigmasq), requires_grad=True)
w_mu = Variable(torch.randn(h, d_mu) * np.sqrt(2/(d+h)), requires_grad=True)
b_mu = Variable(torch.zeros(1, d_mu), requires_grad=True)


def forward(x):
    out = torch.tanh(x.mm(w1.double()) + b1.double())  # shape (n, h)
    #out = F.leaky_relu(x.mm(w1) + b1)  # interesting possibility
    pi = F.softmax(out.mm(w_pi.double()) + b_pi.double(), dim=1)  # p(z_k = 1) for all k; K mixing components that sum to 1; shape (n, k)
    sigmasq = torch.exp(out.mm(w_sigmasq.double()) + b_sigmasq.double())  # K gaussian variances, which must be >= 0; shape (n, k)
    mu = out.mm(w_mu.double()) + b_mu.double()  # K * L gaussian means; shape (n, k*t)

    return pi, sigmasq, mu


def gaussian_pdf(x, mu, sigmasq):
    # NOTE: we could use the new `torch.distributions` package for this now

    return (1/torch.sqrt(2*np.pi*sigmasq).requires_grad_()) * torch.exp((-1/(2*sigmasq)) * torch.norm((x-mu), 2, 1)**2)


def loss_fn(pi, sigmasq, mu, target):
    # PRML eq. 5.153, p. 275
    # compute the likelihood p(y|x) by marginalizing p(z)p(y|x,z)
    # over z. for now, we assume the prior p(w) is equal to 1,
    # although we could also include it here.  to implement this,
    # we average over all examples of the negative log of the sum
    # over all K mixtures of p(z)p(y|x,z), assuming Gaussian
    # distributions.  here, p(z) is the prior over z, and p(y|x,z)
    # is the likelihood conditioned on z and x.

    # print("pi")
    # print(pi)
    # print("sigma")
    # print(sigmasq)
    # print("mu")
    # print(mu)
    # print("target")
    # print(target)

    losses = Variable(torch.zeros(n), requires_grad=True).double()  # p(y|x)
    for i in range(k):  # marginalize over z
        likelihood_z_x = gaussian_pdf(target, mu[:, i*t:(i+1)*t], sigmasq[:, i])
        prior_z = pi[:, i]
        losses += prior_z * likelihood_z_x

    loss = torch.mean(-torch.log(losses))

    return loss

opt = optim.Adam([w1, b1, w_pi, b_pi, w_sigmasq, b_sigmasq, w_mu, b_mu], lr=0.008)

# wrap up the inverse data as Variables
onehot_input, y, _ = cross_entropy_input_to_onehot()
x = Variable(torch.from_numpy(onehot_input).double())
y = Variable(torch.from_numpy(y))

for e in range(3000):
    opt.zero_grad()
    pi, sigmasq, mu = forward(x)
    loss = loss_fn(pi, sigmasq, mu, y)

    if e % 100 == 0:
        print(loss.item())

    loss.backward()
    opt.step()

