from __future__ import annotations
import tensorlib
import typing
import numpy
_Shape = typing.Tuple[int, ...]

__all__ = [
    "Tensor",
    "cpu_add",
    "cpu_bct",
    "cpu_cpy",
    "cpu_div",
    "cpu_exp",
    "cpu_log",
    "cpu_matmul",
    "cpu_mul",
    "cpu_neg",
    "cpu_pow",
    "cpu_relu",
    "cpu_relu_grad",
    "cpu_set_zero",
    "cpu_sub",
    "cpu_sum",
    "cpu_tsp",
    "gpu_add",
    "gpu_bct",
    "gpu_cpy",
    "gpu_div",
    "gpu_exp",
    "gpu_log",
    "gpu_matmul",
    "gpu_mul",
    "gpu_neg",
    "gpu_pow",
    "gpu_relu",
    "gpu_relu_grad",
    "gpu_set_zero",
    "gpu_sub",
    "gpu_sum",
    "gpu_tsp"
]


class Tensor():
    @typing.overload
    def __init__(self, arg0: int, arg1: int) -> None: ...
    @typing.overload
    def __init__(self, arg0: numpy.ndarray[numpy.float32]) -> None: ...
    def __repr__(self) -> str: ...
    def cols(self) -> int: ...
    def cpu(self) -> None: ...
    def data(self) -> float: ...
    def gpu(self) -> None: ...
    def gpuFree(self) -> None: ...
    def numpy(self) -> numpy.ndarray[numpy.float32]: ...
    def onCPU(self) -> bool: ...
    def rows(self) -> int: ...
    pass
def cpu_add(arg0: Tensor, arg1: Tensor) -> Tensor:
    pass
def cpu_bct(arg0: Tensor, arg1: int, arg2: int) -> Tensor:
    pass
def cpu_cpy(arg0: Tensor) -> Tensor:
    pass
def cpu_div(arg0: Tensor, arg1: Tensor) -> Tensor:
    pass
def cpu_exp(arg0: Tensor) -> Tensor:
    pass
def cpu_log(arg0: Tensor) -> Tensor:
    pass
def cpu_matmul(arg0: Tensor, arg1: Tensor) -> Tensor:
    pass
def cpu_mul(arg0: Tensor, arg1: Tensor) -> Tensor:
    pass
def cpu_neg(arg0: Tensor) -> Tensor:
    pass
def cpu_pow(arg0: Tensor, arg1: float) -> Tensor:
    pass
def cpu_relu(arg0: Tensor) -> Tensor:
    pass
def cpu_relu_grad(arg0: Tensor, arg1: Tensor) -> Tensor:
    pass
def cpu_set_zero(arg0: Tensor) -> None:
    pass
def cpu_sub(arg0: Tensor, arg1: Tensor) -> Tensor:
    pass
def cpu_sum(arg0: Tensor, arg1: int) -> Tensor:
    pass
def cpu_tsp(arg0: Tensor) -> Tensor:
    pass
def gpu_add(arg0: Tensor, arg1: Tensor) -> Tensor:
    pass
def gpu_bct(arg0: Tensor, arg1: int, arg2: int) -> Tensor:
    pass
def gpu_cpy(arg0: Tensor) -> Tensor:
    pass
def gpu_div(arg0: Tensor, arg1: Tensor) -> Tensor:
    pass
def gpu_exp(arg0: Tensor) -> Tensor:
    pass
def gpu_log(arg0: Tensor) -> Tensor:
    pass
def gpu_matmul(arg0: Tensor, arg1: Tensor) -> Tensor:
    pass
def gpu_mul(arg0: Tensor, arg1: Tensor) -> Tensor:
    pass
def gpu_neg(arg0: Tensor) -> Tensor:
    pass
def gpu_pow(arg0: Tensor, arg1: float) -> Tensor:
    pass
def gpu_relu(arg0: Tensor) -> Tensor:
    pass
def gpu_relu_grad(arg0: Tensor, arg1: Tensor) -> Tensor:
    pass
def gpu_set_zero(arg0: Tensor) -> None:
    pass
def gpu_sub(arg0: Tensor, arg1: Tensor) -> Tensor:
    pass
def gpu_sum(arg0: Tensor, arg1: int) -> Tensor:
    pass
def gpu_tsp(arg0: Tensor) -> Tensor:
    pass
