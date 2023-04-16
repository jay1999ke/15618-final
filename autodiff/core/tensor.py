import numpy as np
from autodiff import tensorlib
from autodiff.core import exceptions
from autodiff.core.operations import Operations, CPU, GPU


def make_numpy(obj: any):
    t_type = type(obj)
    if t_type == int or t_type == float or t_type == bool:
        obj = np.array([[obj]])
    if isinstance(obj, tensorlib.Tensor):
        return obj
    elif isinstance(obj, np.ndarray):
        pass
    else:
        obj = np.array(obj)
    if len(obj.shape) != 2:
        raise exceptions.AutoDiffException("Invalid object")
    return obj.astype(np.float32)


class Tensor(object):

    def __init__(self, object, requires_grad=False) -> None:
        object = make_numpy(object)
        if isinstance(object, np.ndarray):
            self.value: tensorlib.Tensor = tensorlib.Tensor(object)
        else:
            self.value: tensorlib.Tensor = object
        self.grad: Tensor = None
        self.requires_grad = requires_grad

    def onCPU(self) -> bool:
        return self.value.onCPU()

    def __repr__(self) -> str:
        return self.value.__repr__()

    def cpu(self):
        self.value.cpu()
        if self.grad:
            self.grad.cpu()

    def gpu(self):
        self.value.gpu()
        if self.grad:
            self.grad.gpu()

    def gpuFree(self):
        self.value.gpuFree()
        if self.grad:
            self.grad.gpuFree()

    def __add__(self, other):
        return BinaryOp(self, other, Operations.add)


class BinaryOp(Tensor):

    def __init__(self, a: Tensor, b: Tensor, op: str) -> None:
        aOnCPU = a.onCPU()
        bOnCPU = b.onCPU()
        assert aOnCPU == bOnCPU, "Tensors are not on the same device"
        if aOnCPU:
            value = CPU[op](a.value, b.value)
        else:
            value = GPU[op](a.value, b.value)
        super().__init__(value)
        self.requires_grad = a.requires_grad or b.requires_grad
