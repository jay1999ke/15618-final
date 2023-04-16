import numpy as np
from autodiff import tensorlib
from autodiff.core import exceptions


def make_numpy(obj: any) -> np.ndarray:
    t_type = type(obj)
    if t_type == int or t_type == float or t_type == bool:
        obj = np.array([[obj]])
    elif isinstance(obj, np.ndarray):
        pass
    else:
        obj = np.array(obj)
    if len(obj.shape) != 2:
        raise exceptions.AutoDiffException("Invalid object")
    return obj


class Tensor(object):

    def __init__(self, object, requires_grad = False) -> None:
        object = make_numpy(object)
        self.value: tensorlib.Tensor = tensorlib.Tensor(object)
        self.grad: tensorlib.Tensor = None
        self.requires_grad = requires_grad

    def onCPU(self) -> bool:
        return self.value.onCPU()
    
    def __repr__(self) -> str:
        return self.value.__repr__()
