import numpy as np

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


class CTensor(object):

    def __init__(self, *args) -> None:

        if len(args) == 2:
            self.value = tensorlib.Tensor(*args)
        elif len(args) == 1:
            object = args[0]
            if isinstance(object, tensorlib.Tensor):
                self.value: tensorlib.Tensor = object
            else:
                self.value: tensorlib.Tensor = tensorlib.Tensor(object)

    def __add__(self, other):
        onCpu = onCPU(self.value, other.value)
        if onCpu:
            value = tensorlib.cpu_add(self.value, other.value)
        else:
            value = tensorlib.gpu_add(self.value, other.value)
        return CTensor(value)

    def __mul__(self, other):
        onCpu = onCPU(self.value, other.value)
        if onCpu:
            value = tensorlib.cpu_mul(self.value, other.value)
        else:
            value = tensorlib.gpu_mul(self.value, other.value)
        return CTensor(value)

    def __truediv__(self, other):
        onCpu = onCPU(self.value, other.value)
        if onCpu:
            value = tensorlib.cpu_div(self.value, other.value)
        else:
            value = tensorlib.gpu_div(self.value, other.value)
        return CTensor(value)

    def __rtruediv__(self, other):
        onCpu = onCPU(self.value, other.value)
        if onCpu:
            value = tensorlib.cpu_div(other.value, self.value)
        else:
            value = tensorlib.gpu_div(other.value, self.value)
        return CTensor(value)

    def sum(self, axis: int = 0):
        onCpu = self.onCPU()
        if onCpu:
            value = tensorlib.cpu_sum(self.value, axis)
        else:
            value = tensorlib.gpu_sum(self.value, axis)
        return CTensor(value)

    def pow(self, val: float = 0):
        onCpu = self.onCPU()
        if onCpu:
            value = tensorlib.cpu_pow(self.value, val)
        else:
            value = tensorlib.gpu_pow(self.value, val)
        return CTensor(value)

    def broadcast(self, axis: int, dim: 0):
        onCpu = self.onCPU()
        if onCpu:
            value = tensorlib.cpu_bct(self.value, axis, dim)
        else:
            value = tensorlib.gpu_bct(self.value, axis, dim)
        return CTensor(value)

    def set_zero(self):
        if self.onCPU():
            tensorlib.cpu_set_zero(self.value)
        else:
            tensorlib.gpu_set_zero(self.value)

    def __repr__(self) -> str:
        return self.value.__repr__()

    def copy(self):
        onCpu = self.onCPU()
        if onCpu:
            value = tensorlib.cpu_cpy(self.value)
        else:
            value = tensorlib.gpu_cpy(self.value)
        return CTensor(value)

    def t(self):
        onCpu = self.onCPU()
        if onCpu:
            value = tensorlib.cpu_tsp(self.value)
        else:
            value = tensorlib.gpu_tsp(self.value)
        return CTensor(value)

    def matmul(self, other):
        onCpu = onCPU(self.value, other.value)
        if onCpu:
            value = tensorlib.cpu_matmul(self.value, other.value)
        else:
            value = tensorlib.gpu_matmul(self.value, other.value)
        return CTensor(value)

    def transpose(self):
        return self.t()

    def exp(self):
        onCpu = self.onCPU()
        if onCpu:
            value = tensorlib.cpu_exp(self.value)
        else:
            value = tensorlib.gpu_exp(self.value)
        return CTensor(value)

    def rows(self) -> int:
        return self.value.rows()

    def cols(self) -> int:
        return self.value.cols()

    def onCPU(self) -> bool:
        return self.value.onCPU()

    def cpu(self):
        self.value.cpu()

    def gpu(self):
        self.value.gpu()

    def gpuFree(self):
        self.value.gpuFree()


def onCPU(*tensors: CTensor) -> bool:
    tOnCPU = tensors[0].onCPU()
    for tensor in tensors:
        assert tOnCPU == tensor.onCPU(), "Tensors are not on the same device"
    return tOnCPU
