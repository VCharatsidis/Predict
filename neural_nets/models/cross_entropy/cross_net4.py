"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn


class CrossNet4(nn.Module):
    """
    This class implements a Multi-layer Perceptron in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward.
    """

    def __init__(self, n_inputs):
        """
        Initializes MLP object.

        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP

        TODO:
        Implement initialization of the network.
        """

        super(CrossNet4, self).__init__()

        width = 6
        width_2 = 5
        width_3 = 4
        self.layers = nn.Sequential(

            nn.Linear(n_inputs, width),
            nn.BatchNorm1d(width),
            nn.Tanh(),

            nn.Linear(width, width_2),
            nn.BatchNorm1d(width_2),
            nn.Tanh(),

            nn.Linear(width_2, width_3),
            nn.BatchNorm1d(width_3),
            nn.Tanh(),

            nn.Linear(width_3, width_3),
            nn.BatchNorm1d(width_3),
            nn.Tanh(),

            nn.Linear(width_3, width_3),
            nn.BatchNorm1d(width_3),
            nn.Tanh(),

            nn.Linear(width_3, 3),
            nn.BatchNorm1d(3),
            nn.Tanh(),

            nn.Linear(3, 2),
            nn.BatchNorm1d(2),
            nn.Tanh(),

            nn.Linear(2, 2),
            nn.BatchNorm1d(2),
            nn.Tanh(),

            nn.Linear(2, 1),
            nn.Sigmoid()

        )


    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network

        """

        # out = x
        # for layer in self.layers:
        #     out = layer.forward(out)

        out = self.layers(x)

        return out
