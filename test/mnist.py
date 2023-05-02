import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

# helpers
import numpy as np
import scipy.io as mat

# autodiff
from autodiff.core.tensor import Tensor
from autodiff.neural.layer import Linear
from autodiff.neural.loss import MeanSquareError, NLLLoss
from autodiff.neural.optimizer import Optimizer




def get_accuracy_value(pred: Tensor, np_y: np.ndarray):
    pred_copy: Tensor = pred.copy()
    pred_copy.cpu()
    pred_copy.gpuFree()
    np_pred: np.ndarray = pred_copy.numpy()

    np_pred: np.ndarray = np_pred.argmax(1)
    np_y: np.ndarray = np_y.argmax(1)
    return (np_pred == np_y).mean()

if __name__ == "__main__":
    raw_data = mat.loadmat("test/data/mnist_reduced.mat")
    X = raw_data['X'].astype(np.float32)
    y = raw_data['y'].ravel()

    Y = np.zeros((5000, 10), dtype='uint8')
    for i, row in enumerate(Y):
        Y[i, y[i] - 1] = 1
    y = Y.astype(np.float32)
    np_y = y

    X = Tensor(X)
    y = Tensor(y)
    X.gpu()
    y.gpu()

    layer1 = Linear(400, 25)
    layer1.bias.gpu()
    layer1.weight.gpu()
    layer2 = Linear(25, 25)
    layer2.bias.gpu()
    layer2.weight.gpu()
    layer3 = Linear(25, 10)
    layer3.bias.gpu()
    layer3.weight.gpu()

    lr = 5e-2
    optim = Optimizer(learning_rate=lr)

    for x in range(20000):

        l1 = layer1(X).relu()
        l2 = layer2(l1).relu()
        l3 = layer3(l2).log_softmax(axis=1)
        loss = NLLLoss(l3, y)
        loss.backward()
        optim.step(loss)
        loss.zero_grad()
        
        if x % 50 == 0:
            print(x, get_accuracy_value(l3, np_y), loss)