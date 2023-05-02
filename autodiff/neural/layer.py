import numpy as np
from torch import zero_

import autodiff.functional as F
from autodiff.core.tensor import Tensor


class Weight(Tensor):
    """The datastructure that holds parameters of a Model"""

    def __init__(self, shape, **kwargs):
        # Initialize the weights using the Kaiming initialization method
        nparray = np.random.randn(*shape) * np.sqrt(2 / shape[0])
        if kwargs.get("zero") == True:
            nparray *= 0
        super().__init__(object=nparray)
        self.requires_grad = True
        self.iteration: int = 0

    def update_weights(self, lr: float, iteration: int):
        if iteration > self.iteration: # update once per iteration
            lr: Tensor = Tensor(-lr)
            if not self.onCPU():
                lr.gpu()
            self.value = self.value + (lr * self.grad).value
            self.iteration = iteration

class Layer(object):

    def __call__(self):
        raise NotImplementedError

class Linear(Layer):

    def __init__(self, in_dim, out_dim, bias=True):
        self.weight = Weight(shape=(in_dim, out_dim))
        if bias:
            self.bias = Weight(shape=(1, out_dim))
        self.bias_present = bias

    def __call__(self, inputs: Tensor):
        if self.bias_present:
            return F.MatMul(inputs, self.weight) + self.bias
        else:
            return F.MatMul(inputs, self.weight)
