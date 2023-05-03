import sys, os

import torch
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

# helpers
import numpy as np
import scipy.io as mat

# autodiff
from torch import nn
from torch import Tensor
from torch.nn import NLLLoss
import torch.optim as optim
import torch.nn.functional as F


def get_accuracy_value(pred: Tensor, np_y: np.ndarray):
    pred_copy: Tensor = pred.clone()
    pred_copy = pred_copy.cpu()
    np_pred: np.ndarray = pred_copy.detach().numpy()

    np_pred: np.ndarray = np_pred.argmax(1)
    np_y: np.ndarray = np_y.argmax(1)
    return (np_pred == np_y).mean()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(400, 25)
        self.layer2 = nn.Linear(25, 25)
        self.layer3 = nn.Linear(25, 10)

    def forward(self, x):
        x = self.layer1(x).relu()
        x = self.layer2(x).relu()
        x = self.layer3(x).log_softmax(dim=1)
        return x

if __name__ == "__main__":
    raw_data = mat.loadmat("test/data/mnist_reduced.mat")
    X = raw_data['X'].astype(np.float32)
    y = raw_data['y'].ravel()
    org_y = y - 1

    Y = np.zeros((5000, 10), dtype='uint8')
    for i, row in enumerate(Y):
        Y[i, y[i] - 1] = 1
    y = Y.astype(np.float32)
    np_y = y

    X = Tensor(X)
    y = Tensor(y)
    org_y = Tensor(org_y).type(torch.long)
    X = X.cuda()
    org_y = org_y.cuda()

    model = Net()
    model = model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=5e-2)

    for x in range(10000):
        l3 = model(X)
        loss = F.nll_loss(l3, org_y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if x % 50 == 0:
            print(x, get_accuracy_value(l3, np_y), loss.item())