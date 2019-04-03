import numpy as np
from layers import *
    

class ConvNet(object):
    """
    A convolutional network.

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(1, 28, 28), num_classes=10):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_classes: Number of scores to produce from the final affine layer.
        """
        self.params = {}

        #######################################################################
        # TODO: Initialize weights and biases for the convolutional neural    #
        # network. Weights should be initialized from a Gaussian distribution;#
        # biases should be initialized to zero. All weights and biases should #
        # be stored in the dictionary self.params.                            #
        #######################################################################
        (C, H, W) = input_dim
        # Conv1
        self.params['W1'] = np.random.randn(6, C, 5, 5) * 0.01
        self.params['b1'] = np.zeros(6)
        # # Conv2
        # self.params['W2'] = np.random.randn(16, C, 5, 5)
        # self.params['b2'] = np.zeros(16)

        # fc1
        temp = int(6 * H * W)
        self.params['W2'] = np.random.randn(temp, num_classes) * 0.01
        self.params['b2'] = np.zeros(num_classes)

        #######################################################################
        #                         END OF YOUR CODE                            #
        #######################################################################

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        N = X.shape[0]
        mode = 'test' if y is None else 'train'
        scores = None

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        #######################################################################
        # TODO: Implement the forward pass for the convolutional neural net,  #
        # computing the class scores for X and storing them in the scores     #
        # variable.                                                           #
        #######################################################################
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}
        out1, cache1 = conv_forward(X, W1, b1, conv_param)
        # print("out1 = ", out1[0])
        
        # pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        # out2, cache2 = max_pool_forward(out1, pool_param)
        
        out3, cache3 = relu_forward(out1)
        # print("out3 = ", out3[0])
        
        scores, cache4 = fc_forward(out3, W2, b2)
        # print("scores = ", scores[0])
        
        #######################################################################
        #                             END OF YOUR CODE                        #
        #######################################################################
        if y is None:
            return scores

        loss, grads = 0, {}
        #######################################################################
        # TODO: Implement the backward pass for the convolutional neural net, #
        # storing the loss and gradients in the loss and grads variables.     #
        # Compute data loss using softmax, and make sure that grads[k] holds  #
        # the gradients for self.params[k].                                   #
        #######################################################################
        loss, dscores = softmax_loss(scores, y)

        dout3, dW2, db2 = fc_backward(dscores, cache4)
        
        dout1 = relu_backward(dout3, cache3)
        
        # dout1 = max_pool_backward(dout2, cache2)
        
        dX, dW1, db1 = conv_backward(dout1, cache1)
        
        grads['W1'] = dW1
        grads['W2'] = dW2
        grads['b1'] = db1
        grads['b2'] = db2

        #######################################################################
        #                             END OF YOUR CODE                        #
        #######################################################################

        return loss, grads
