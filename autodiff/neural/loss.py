from autodiff.core.tensor import Tensor, GraphNode
import autodiff.functional as F

def MeanSquareError(pred: Tensor, target: Tensor) -> Tensor:
    assert pred.shape == target.shape
    mse = ((pred - target) ** 2).mean()
    return mse


def NLLLoss(pred: Tensor, target: Tensor):
    assert pred.shape == target.shape
    return - F.Mean(F.Sum(pred * target, axis = 1), axis = 0)