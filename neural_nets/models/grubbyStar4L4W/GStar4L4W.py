"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn


class GStar4L4WNet(nn.Module):
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

        super(GStar4L4WNet, self).__init__()

        width = 4
        width_2 = 20
        self.layers = nn.Sequential(

            nn.Linear(n_inputs, width),
            nn.BatchNorm1d(width),
            nn.Tanh(),

            nn.Linear(width, width_2),
            nn.BatchNorm1d(width_2),
            nn.Tanh(),

            nn.Linear(width_2, width_2 // 2),
            nn.BatchNorm1d(width_2 // 2),
            nn.Tanh(),

            nn.Linear(width_2 // 2, width_2 // 2),
            nn.BatchNorm1d(width_2 // 2),
            nn.Tanh(),

            nn.Linear(width_2 // 2, width_2 // 2),
            nn.BatchNorm1d(width_2 // 2),
            nn.Tanh(),

            nn.Linear(width_2 // 2, width_2 // 2),
            nn.BatchNorm1d(width_2 // 2),
            nn.Tanh(),

            nn.Linear(width_2 // 2, width_2 // 2),
            nn.BatchNorm1d(width_2 // 2),
            nn.Tanh(),

            nn.Linear(width_2 // 2, width_2 // 4),
            nn.BatchNorm1d(width_2 // 4),
            nn.Tanh(),

            nn.Linear(width_2 // 4, width_2 // 4),
            nn.BatchNorm1d(width_2 // 4),
            nn.Tanh(),

            nn.Linear(width_2 // 4, width_2 // 4),
            nn.BatchNorm1d(width_2 // 4),
            nn.Tanh(),

            nn.Linear(width_2 // 4, 1),
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

        TODO:
        Implement forward pass of the network.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################

        # out = x
        # for layer in self.layers:
        #     out = layer.forward(out)

        out = self.layers(x)
        ########################
        # END OF YOUR CODE    #
        #######################

        return out