import numpy as np
from typing import Tuple, List

from autodiff import tensorlib
from autodiff.core import exceptions


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
        self._requires_grad = False
        self.shape: Tuple[int, int] = self.value.rows(), self.value.cols()
        self.requires_grad = requires_grad

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad
    
    @requires_grad.setter
    def requires_grad(self, val: bool) -> None:
        if val:
            self._requires_grad = True
            if not self.grad:
                self.grad = tensorlib.Tensor(*self.shape)
                self.zero_grad()
        else:
            self.grad: tensorlib.Tensor = None
        

    def onCPU(self) -> bool:
        return self.value.onCPU()
    
    def zero_grad(self) -> None:
        if self._requires_grad:
            if self.onCPU():
                tensorlib.cpu_set_zero(self.grad)
            else:
                tensorlib.gpu_set_zero(self.grad)

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
        return Add(self, other)

    def __mul__(self, other):
        return Multiply(self, other)

    def sum(self, axis: int = 0):
        return Sum(self, axis)

    def broadcast(self, axis: int, dim: 0):
        return Broadcast(self, axis, dim)


def onCPU(*tensors: Tensor) -> bool:
    tOnCPU = tensors[0].onCPU()
    for tensor in tensors:
        assert tOnCPU == tensor.onCPU(), "Tensors are not on the same device"
    return tOnCPU


def _broadcast(a: Tensor, b: Tensor) -> Tuple[Tensor, Tensor]:
    """Broadcast to match shape"""
    if a.shape == b.shape:
        pass
    elif a.shape[0] == b.shape[0]:
        if a.shape[1] == 1:
            a = a.broadcast(axis=1, dim=b.shape[1])
        elif b.shape[1] == 1:
            b = b.broadcast(axis=1, dim=a.shape[1])
        else:
            raise exceptions.ShapeMismatchException("Shapes don't match")
    elif a.shape[1] == b.shape[1]:
        if a.shape[0] == 1:
            a = a.broadcast(axis=0, dim=b.shape[0])
        elif b.shape[0] == 1:
            b = b.broadcast(axis=0, dim=a.shape[0])
        else:
            raise exceptions.ShapeMismatchException("Shapes don't match")
    elif a.shape[0] == (1, 1):
        a = a.broadcast(axis=0, dim=b.shape[0])
        a = a.broadcast(axis=1, dim=b.shape[1])
    elif b.shape == (1, 1):
        b = b.broadcast(axis=0, dim=a.shape[0])
        b = b.broadcast(axis=1, dim=a.shape[1])
    else:
        raise exceptions.ShapeMismatchException("Shapes don't match")
    return a, b


class Add(Tensor):

    def __init__(self, a: Tensor, b: Tensor) -> None:
        onCpu = onCPU(a, b)
        a, b = _broadcast(a, b)
        if onCpu:
            value = tensorlib.cpu_add(a.value, b.value)
        else:
            value = tensorlib.gpu_add(a.value, b.value)
        super().__init__(value)
        self.requires_grad = a.requires_grad or b.requires_grad


class Multiply(Tensor):

    def __init__(self, a: Tensor, b: Tensor) -> None:
        onCpu = onCPU(a, b)
        a, b = _broadcast(a, b)
        if onCpu:
            value = tensorlib.cpu_mul(a.value, b.value)
        else:
            value = tensorlib.gpu_mul(a.value, b.value)
        super().__init__(value)
        self.requires_grad = a.requires_grad or b.requires_grad


class Sum(Tensor):

    def __init__(self, a: Tensor, axis: int) -> None:
        onCpu = onCPU(a)
        if onCpu:
            value = tensorlib.cpu_sum(a.value, axis)
        else:
            value = tensorlib.gpu_sum(a.value, axis)
        super().__init__(value)
        self.requires_grad = a.requires_grad


class Broadcast(Tensor):

    def __init__(self, a: Tensor, axis: int, dim: int) -> None:
        onCpu = onCPU(a)
        if onCpu:
            value = tensorlib.cpu_bct(a.value, axis, dim)
        else:
            value = tensorlib.gpu_bct(a.value, axis, dim)
        super().__init__(value)
        self.requires_grad = a.requires_grad
