import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive

def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])  #[10, 5, 10]
    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))               #[10,5]
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))                            #[1, 5]
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))              #[5*10]
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))                       #[1, 10]

    #data = [20, 10]
    hInp = np.dot(data, W1) + b1   #l1 [20, 5]
    hOp = sigmoid(hInp)                 #h   [20, 5]
    OInp = np.dot(hOp, W2) + b2  #l2  [20, 10]
    op = softmax(OInp)                  #      [20, 10]
        ### END YOUR CODE

        ### YOUR CODE HERE: backward propagation
        #raise NotImplementedError:
    cost = -np.sum(labels * np.log(op))  /data.shape[0]  #is it a scalar

    delO = op - labels                                                                   #[20, 10]
    delW2 = np.dot(hOp.T, delO)                                              #[5, 10]  = [20, 5]' * [20, 10]
    delB2 = np.dot(np.ones((1, data.shape[0])), delO)       #[1, 10] = [1, 20] * [20, 10]

    delHOp = np.dot(delO, W2.T)                                             #[20, 10] * [5, 10]' = [20, 5]
    delHInp = delHOp * sigmoid_grad(hOp)                         #[20, 5]
    delW1 = np.dot(data.T, delHInp)
    delB1 = np.dot(np.ones((1, data.shape[0])), delHInp)
    ### END YOUR CODE

    gradW2 = delW2/data.shape[0]
    gradb2 = delB2/data.shape[0]
    gradW1 = delW1/data.shape[0]
    gradb1 = delB1/data.shape[0]     #'''
    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad

def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    #
    #generating some data
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    #
    #Adding labels detail
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i,random.randint(0,dimensions[2]-1)] = 1
    #
    #all the Ws, including the bias term
    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )
    #
    #
    gradcheck_naive(lambda params: forward_backward_prop(data, labels, params,
        dimensions), params)

def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    #raise NotImplementedError
    ### END YOUR CODE

if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
