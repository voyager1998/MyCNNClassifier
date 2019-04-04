import numpy as np
from sklearn.datasets import fetch_openml
import os
import pickle

# from network import ConvNet
from conv_2_network import ConvNet
from solver import Solver

def save_obj(obj, name ):
    with open('../data/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('../data/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def load_fashion_mnist(flatten=False):
    data = {}
    mnist = fetch_openml(name="Fashion-MNIST")
    X = mnist.data
    y = mnist.target.astype(np.uint8)

    #######################################################################
    # Optional: you're free to preprocess images here                     #
    #######################################################################
    pass
    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################

    if not flatten:
        X = X.reshape(X.shape[0], 28, 28)
        X = X[:, np.newaxis, :, :]
    print(X.shape)
    print(y.shape)    
    #######################################################################
    # Optional: you're free to adjust the training and val split.         #
    # However, the last 10000 images must be the test set.                #
    #######################################################################
    data['X_train'] = X[:50000]
    data['y_train'] = y[:50000]
    data['X_val'] = X[50000:60000]
    data['y_val'] = y[50000:60000]
    data['X_test'] = X[60000:]
    data['y_test'] = y[60000:]
    return data


def train():
    # load data
    if os.path.exists("../data/MNIST.pkl"):
        data = load_obj("MNIST")
        print("data loaded!")
    else:
        data = load_fashion_mnist()
        print("data downloaded!")
        save_obj(data, "MNIST")
        print("data saved!")

    # initialize model
    print("initialize model")
    model = ConvNet()

    # intialize solver
    print("initialize solver")
    solver = Solver(model, data, update_rule='sgd',
                    optim_config={'learning_rate': 1e-4},
                    lr_decay=1.0, num_epochs=10,
                    batch_size=16, print_every=1)

    # start training
    print("start training")    
    solver.train()

    # report test accuracy
    print("report test accuracy")
    acc = solver.check_accuracy(data['X_test'], data['y_test'])
    print("Test accuracy: {}".format(acc))


if __name__=="__main__":
    train()
