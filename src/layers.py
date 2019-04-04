import numpy as np
from scipy import special
from scipy import signal
import math

def fc_forward(x, w, b):
    """
    Computes the forward pass for a fully-connected layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    cache = (x, w, b)
    ###########################################################################
    # Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    N = x.shape[0]
    out = np.reshape(x, (N, -1)) # out: N * D
    D = out.shape[1]
    out = np.matmul(out, w) + b # out: N * M
    # for i in range(len(out)):
    #   out[i] += b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def fc_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d_1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = x, w, b
    N = x.shape[0]
    ###########################################################################
    # Implement the affine backward pass.                               #
    ###########################################################################
    dx = np.matmul(dout, np.transpose(w))
    dx = np.reshape(dx, x.shape)
    x = np.reshape(x, (N, -1)) # out: N * D
    dw = np.matmul(np.transpose(x), dout)
    db = np.sum(dout, axis=0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    ###########################################################################
    # Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = x * (x > 0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # Implement the ReLU backward pass.                                 #
    ###########################################################################
    dRdx = x > 0
    dx = dout.reshape(x.shape) * dRdx
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def conv(x, kernel, padding, stride=1):
    x_pad = np.pad(x, ((padding, padding), (padding, padding)), 'constant', constant_values=0)
    out = signal.correlate(x_pad, kernel, mode='valid', method='fft')
    n_rows = math.ceil(len(out) / stride)
    n_cols = math.ceil(len(out[0]) / stride)
    result = np.zeros((n_rows, n_cols))
    for i in range(n_rows):
        for j in range(n_cols):
            result[i][j] = out[i * stride][j * stride]
    return result

def conv_forward(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.


    During padding, 'pad' zeros should be placed symmetrically (i.e equally on
    both sides) along the height and width axes of the input. Be careful not to
    modfiy the original input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    cache = (x, w, b, conv_param)
    stride = conv_param['stride']
    pad = conv_param['pad']
    (N, C, H, W) = x.shape
    (F, C, HH, WW) = w.shape
    Hp = int(1 + (H + 2 * pad - HH) / stride)
    Wp = int(1 + (W + 2 * pad - WW) / stride)
    out = np.zeros((N, F, Hp, Wp))
    ###########################################################################
    # Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    for i in range(N):
      for j in range(F):
        for c in range(C):
          out[i][j] += conv(x[i][c], w[j][c], pad, stride)
        out[i][j] += b[j]
        
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache

def dconv(dout, kernel, padding, stride=1):
    ker_rev = np.flip(kernel)
    H, W = dout.shape
    HH, WW = kernel.shape
    dout_ext = np.zeros((stride * (H - 1) + 1, stride * (W - 1) + 1))
    for i in range(H):
      for j in range(W):
        dout_ext[stride * i][stride * j] = dout[i][j]
    pad_H = HH - 1 - padding
    pad_W = WW - 1 - padding
    dout_ext = np.pad(dout_ext, ((pad_H, pad_H), (pad_W, pad_W)), 'constant', constant_values=0)
    dx = conv(dout_ext, ker_rev, 0, 1)
    return dx

def dwconv(x, dout, padding, stride=1):
    x_ext = np.pad(x, ((padding, padding), (padding, padding)), 'constant', constant_values=0)
    H, W = x.shape
    HH, WW = dout.shape
    dout_ext=np.zeros((stride * (HH - 1) + 1, stride * (WW - 1) + 1))
    for i in range(HH):
      for j in range(WW):
        dout_ext[stride * i][stride * j] = dout[i][j]
    dw = conv(x, dout_ext, 0, 1)
    return dw

def conv_backward(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    (x, w, b, conv_param) = cache
    stride = conv_param['stride']
    pad = conv_param['pad']
    (N, C, H, W) = x.shape
    (F, C, HH, WW) = w.shape
    (N, F, Hp, Wp) = dout.shape
    ###########################################################################
    # Implement the convolutional backward pass.                        #
    ###########################################################################
    dx = np.zeros(x.shape)
    dw = np.zeros(w.shape)
    db = np.sum(dout, axis=(0,2,3))
    
    for i in range(N):
      for j in range(F):
        for c in range(C):
          dx[i][c] += dconv(dout[i][j], w[j][c], pad, stride)
          dw[j][c] += dwconv(x[i][c], dout[i][j], pad, stride)
    
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    cache = (x, pool_param)
    ###########################################################################
    # Implement the max-pooling forward pass                            #
    ###########################################################################
    (N, C, H, W) = x.shape
    Hp = 1 + int((H - pool_param['pool_height']) / pool_param['stride'])
    Wp = 1 + int((W - pool_param['pool_width']) / pool_param['stride'])
    out = np.zeros((N, C, Hp, Wp))
    for i in range(N):
      for j in range(C):
        for m in range(Hp):
          for n in range(Wp):
            start_h = m * pool_param['stride']
            start_w = n * pool_param['stride']
            out[i][j][m][n] = np.amax(x[i, j, start_h: start_h + pool_param['pool_height'], \
               start_w: start_w + pool_param['pool_width']])
            
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def max_pool_backward(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    (x, pool_param) = cache
    ###########################################################################
    # Implement the max-pooling backward pass                           #
    ###########################################################################
    dx = np.zeros(x.shape)
    (N, C, Hp, Wp) = dout.shape
    for i in range(N):
      for j in range(C):
        for m in range(Hp):
          for n in range(Wp):
            start_h = m * pool_param['stride']
            start_w = n * pool_param['stride']
            pos = np.argmax(x[i, j, start_h: start_h + pool_param['pool_height'], \
               start_w: start_w + pool_param['pool_width']])
            r = int(pos / pool_param['pool_width'])
            c = int(pos - r * pool_param['pool_width'])
            dx[i][j][start_h + r][start_w + c] += dout[i][j][m][n]
            
            

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def l2_loss(x, y):
    """
    Computes the loss and gradient of L2 loss.
    loss = 1/N * sum((x - y)**2)

    Inputs:
    - x: Input data, of shape (N, D)
    - y: Output data, of shape (N, D)

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = 0, None
    ###########################################################################
    # Implement L2 loss                                                 #
    ###########################################################################
    N, D = x.shape
    for i in range(N):
      for j in range(D):
        loss += (x[i][j] - y[i][j]) ** 2
    loss /= N
    dx = 2 / N * (x - y)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = 0, None
    ###########################################################################
    # Implement softmax loss                                     #
    ###########################################################################
    N = x.shape[0]
    
    sm = special.softmax(x, axis = 1)
    sm += np.finfo(float).eps
    for i in range(N):
      loss += np.log(sm[i][y[i]])
    loss = -1 / N * loss
    dx = sm
    dx[range(N), y] -= 1
    dx = dx / N
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


if __name__ == "__main__":
  print("test for fc_forward:")
  x = np.array( \
    [[[0., 1., 2., 3.],\
    [4., 5., 6., 7.],\
    [0., 0., 0., 0.]],\
    [[0., 0., 0., 0.],\
    [0., 0., 0., 0.],\
    [0., 0., 0., 0.]]])
  w = np.ones((12, 3))
  b = np.array([1, 2, 3])
  out, _ = fc_forward(x, w, b)
  print(out)
  print("------------------------------------")

  print("test for Loss:")
  x = np.array([1, 2, 3])
  y = np.array([0, 1, 4])
  print(l2_loss(x,y))
  print("------------------------------------")

  print("test for Softmax_loss:")
  x = np.array([[1, 2, 3],[0, 1, 4]])
  y = np.array([1, 2])
  print(softmax_loss(x, y))
  print("------------------------------------")
