from __future__ import annotations
import tensorlib
import typing
import numpy
_Shape = typing.Tuple[int, ...]

__all__ = [
    "Tensor",
    "cpu_add",
    "cpu_mul",
    "cpu_sum",
    "gpu_add",
    "gpu_mul",
    "gpu_sum"
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
    def onCPU(self) -> bool: ...
    def rows(self) -> int: ...
    pass
def cpu_add(arg0: Tensor, arg1: Tensor) -> Tensor:
    pass
def cpu_mul(arg0: Tensor, arg1: Tensor) -> Tensor:
    pass
def cpu_sum(arg0: Tensor, arg1: int) -> Tensor:
    pass
def gpu_add(arg0: Tensor, arg1: Tensor) -> Tensor:
    pass
def gpu_mul(arg0: Tensor, arg1: Tensor) -> Tensor:
    pass
def gpu_sum(arg0: Tensor, arg1: int) -> Tensor:
    pass
