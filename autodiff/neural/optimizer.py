from typing import List
from autodiff.core.tensor import Tensor, GraphNode
from autodiff.neural.layer import Weight

class Optimizer(object):

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.iteration = 0

    def step(self, loss: Tensor):
        self.update_weights(loss.parents)
        self.iteration += 1
        if self.iteration % 10000 == 0:
            self.learning_rate *= 0.91

    def update_weights(self, parents: List[GraphNode]):
        if parents:
            for parent in parents:
                if isinstance(parent.var, Weight):
                    parent.var.update_weights(self.learning_rate, self.iteration)
            grandParents: List[GraphNode] = []
            for parent in parents:
                grandParents.extend(parent.var.parents)
            self.update_weights(grandParents)
        
