import numpy as np
from typing import Tuple, List

from autodiff.core import exceptions
from autodiff.core.internal import CTensor


def make_numpy(obj: any):
    t_type = type(obj)
    if t_type == int or t_type == float or t_type == bool:
        obj = np.array([[obj]])
    if isinstance(obj, CTensor):
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
        if isinstance(object, CTensor):
            self.value: CTensor = object
        else:
            object = make_numpy(object)
            self.value: CTensor = CTensor(object)
        self.grad: Tensor = None
        self._requires_grad = False
        self.shape: Tuple[int, int] = self.value.rows(), self.value.cols()
        self.requires_grad = requires_grad
        # Store topological lineage locally
        self.parents: List[GraphNode] = []

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, val: bool) -> None:
        if val:
            self._requires_grad = True
            if not self.grad:
                self.grad: Tensor = Tensor(
                    CTensor(*self.shape), requires_grad=False
                )
                if not self.value.onCPU():
                    self.grad.gpu()  # move grad to gpu is value is on gpu
                self.zero_grad()
        else:
            self.grad: Tensor = None

    def onCPU(self) -> bool:
        onCpu = self.value.onCPU()
        if self.grad:
            assert onCpu == self.grad.onCPU()
        return onCpu

    def zero_grad(self) -> None:
        if self._requires_grad:
            self.grad.value.set_zero()
        for parent in self.parents:
            parent.var.zero_grad()

    def __repr__(self) -> str:
        return self.value.__repr__()

    def copy(self, copy_grad: bool = False):
        value = self.value.copy()
        tensor = Tensor(value)
        if copy_grad and self.grad:
            tensor.grad = self.grad.copy()
        return tensor

    def numpy(self):
        return self.value.numpy()

    def cpu(self):
        self.value.cpu()
        self.value.gpuFree()  # until this is fixed in tensorlib
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

    def __sub__(self, other):
        return Subtract(self, other)

    def __rsub__(self, other):
        return Subtract(other, self)

    def __neg__(self):
        return Negate(self)

    def __mul__(self, other):
        return Multiply(self, other)

    def __truediv__(self, other):
        return Divide(self, other)

    def __rtruediv__(self, other):
        return Divide(other, self)

    def sum(self, axis: int = 0):
        return Sum(self, axis)

    def max(self, axis: int):
        return Max(self, axis)

    def exp(self):
        return Exp(self)

    def log(self):
        return Log(self)

    def mm(self, other):
        return MatMul(self, other)

    def matmul(self, other):
        return self.mm(other)

    def t(self):
        return Transpose(self)

    def __pow__(self, val: float):
        return Power(self, val)

    def pow(self, val: float):
        return Power(self, val)

    def transpose(self):
        return self.t()

    def broadcast(self, axis: int, dim: 0):
        return Broadcast(self, axis, dim)

    def sigmoid(self):
        return Sigmoid(self)

    def relu(self):
        return Relu(self)

    def log_softmax(self, axis: int):
        return LogSoftmax(self, axis)

    def mean(self, axis: int = -1):
        return Mean(self, axis=axis)

    def backward(self, gradient=None) -> None:
        """Propagates appropriate gradient to 
        local reverse computational sub-graph"""

        if not self.requires_grad:
            raise exceptions.AutoDiffException(
                "Gradient called on a non-differentiable variable")

        if gradient is None:
            if self.shape == (1, 1):
                gradient = Tensor(1)
                if not self.onCPU():
                    gradient.gpu()
            else:
                raise exceptions.AutoDiffException("Gradient not provided")

        if not isinstance(gradient, Tensor):
            raise exceptions.AutoDiffException(
                "Gradient is not of type Tensor")

        assert gradient.shape == self.shape, f"Gradient shape mismatch: {gradient.shape}, {self.shape}"

        assert gradient.requires_grad == False, "Recursion hell?"

        # swap old gradient internal tensor with new one
        # tensorlib does not support inplace operations
        self.grad.value = (self.grad + gradient).value

        for parent in self.parents:
            parent.gradient_prop(gradient)


class GraphNode(object):
    """A Node on a reverse computation graph"""

    def __init__(self, tensor: Tensor, vjp):
        self.var: Tensor = tensor
        self.vjp = vjp

    def gradient_prop(self, gradient: Tensor):
        assert gradient.requires_grad == False, "Recursion hell?"
        if self.var.requires_grad:
            self.var.backward(self.vjp(gradient))


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
    elif a.shape[0] == 1:  # potential broadcast (TODO: check correctness)
        a = a.broadcast(axis=0, dim=b.shape[0])
        a, b = _broadcast(a, b)
    elif b.shape[0] == 1:  # potential broadcast (TODO: check correctness)
        b = b.broadcast(axis=0, dim=a.shape[0])
        a, b = _broadcast(a, b)
    else:
        raise exceptions.ShapeMismatchException("Shapes don't match")
    return a, b


class Add(Tensor):

    def __init__(self, a: Tensor, b: Tensor) -> None:
        a, b = _broadcast(a, b)
        value = a.value + b.value
        super().__init__(value)
        self.requires_grad = a.requires_grad or b.requires_grad

        if a.requires_grad:
            def vjp_a(gradient: Tensor) -> Tensor:
                return gradient
            self.parents.append(GraphNode(tensor=a, vjp=vjp_a))

        if b.requires_grad:
            def vjp_b(gradient: Tensor) -> Tensor:
                return gradient
            self.parents.append(GraphNode(tensor=b, vjp=vjp_b))


class Subtract(Tensor):

    def __init__(self, a: Tensor, b: Tensor) -> None:
        a, b = _broadcast(a, b)
        value = a.value - b.value
        super().__init__(value)
        self.requires_grad = a.requires_grad or b.requires_grad

        if a.requires_grad:
            def vjp_a(gradient: Tensor) -> Tensor:
                return gradient
            self.parents.append(GraphNode(tensor=a, vjp=vjp_a))

        if b.requires_grad:
            def vjp_b(gradient: Tensor) -> Tensor:
                return -gradient
            self.parents.append(GraphNode(tensor=b, vjp=vjp_b))


class Negate(Tensor):

    def __init__(self, a: Tensor) -> None:
        super().__init__(-a.value)
        self.requires_grad = a.requires_grad

        if a.requires_grad:
            def vjp_a(gradient: Tensor) -> Tensor:
                return -gradient
            self.parents.append(GraphNode(tensor=a, vjp=vjp_a))


class Multiply(Tensor):

    def __init__(self, a: Tensor, b: Tensor) -> None:
        a, b = _broadcast(a, b)
        value = a.value * b.value
        super().__init__(value)
        self.requires_grad = a.requires_grad or b.requires_grad

        if a.requires_grad:
            def vjp_a(gradient: Tensor) -> Tensor:
                return Tensor(gradient.value * b.value)
            self.parents.append(GraphNode(tensor=a, vjp=vjp_a))

        if b.requires_grad:
            def vjp_b(gradient: Tensor) -> Tensor:
                return Tensor(gradient.value * a.value)
            self.parents.append(GraphNode(tensor=b, vjp=vjp_b))


class Divide(Tensor):

    def __init__(self, a: Tensor, b: Tensor) -> None:
        a, b = _broadcast(a, b)
        value = a.value / b.value
        super().__init__(value)
        self.requires_grad = a.requires_grad or b.requires_grad

        if a.requires_grad:
            def vjp_a(gradient: Tensor) -> Tensor:
                return Tensor(gradient.value / b.value)
            self.parents.append(GraphNode(tensor=a, vjp=vjp_a))

        if b.requires_grad:
            def vjp_b(gradient: Tensor) -> Tensor:
                negative = Tensor(-1)
                if not a.onCPU():
                    negative.gpu()
                part = negative * Tensor(a.value / b.value.pow(2))
                return Tensor(gradient.value * part.value)
            self.parents.append(GraphNode(tensor=b, vjp=vjp_b))


class Sum(Tensor):

    def __init__(self, a: Tensor, axis: int) -> None:
        super().__init__(a.value.sum(axis=axis))
        self.requires_grad = a.requires_grad

        if a.requires_grad:
            assert axis < 2, "Axis greater than 1"
            dim = a.shape[axis]

            def vjp(gradient: Tensor) -> Tensor:
                gradient = gradient.value.broadcast(axis=axis, dim=dim)
                return Tensor(gradient)
            self.parents.append(GraphNode(tensor=a, vjp=vjp))


class Broadcast(Tensor):

    def __init__(self, a: Tensor, axis: int, dim: int) -> None:
        super().__init__(a.value.broadcast(axis=axis, dim=dim))
        self.requires_grad = a.requires_grad

        if a.requires_grad:
            def vjp(gradient: Tensor) -> Tensor:
                gradient = gradient.value.sum(axis=axis)
                return Tensor(gradient)
            self.parents.append(GraphNode(tensor=a, vjp=vjp))


class Exp(Tensor):

    def __init__(self, a: Tensor) -> None:
        super().__init__(a.value.exp())
        self.requires_grad = a.requires_grad

        if a.requires_grad:
            def vjp(gradient: Tensor) -> Tensor:
                return Tensor(gradient.value * self.value)
            self.parents.append(GraphNode(tensor=a, vjp=vjp))


class Log(Tensor):

    def __init__(self, a: Tensor) -> None:
        super().__init__(a.value.log())
        self.requires_grad = a.requires_grad

        if a.requires_grad:
            def vjp(gradient: Tensor) -> Tensor:
                return Tensor(gradient.value / a.value)
            self.parents.append(GraphNode(tensor=a, vjp=vjp))


class Transpose(Tensor):

    def __init__(self, a: Tensor) -> None:
        super().__init__(a.value.t())
        self.requires_grad = a.requires_grad

        if a.requires_grad:
            def vjp(gradient: Tensor) -> Tensor:
                return Tensor(gradient.value.t())
            self.parents.append(GraphNode(tensor=a, vjp=vjp))


class Power(Tensor):
    # only int supported right now, TODO: Add tensor power
    # all funcs using broadcast should have 2 modes: scaler and tensor

    def __init__(self, a: Tensor, val: float) -> None:
        super().__init__(a.value.pow(val))
        self.requires_grad = a.requires_grad

        if a.requires_grad:
            def vjp(gradient: Tensor) -> Tensor:
                onCpu = gradient.onCPU()
                valTensor = Tensor(val)
                if not onCpu:
                    valTensor.gpu()
                return Tensor(gradient.value * (a.value.pow(val - 1))) * valTensor
            self.parents.append(GraphNode(tensor=a, vjp=vjp))


class MatMul(Tensor):

    def __init__(self, a: Tensor, b: Tensor) -> None:
        super().__init__(a.value.matmul(b.value))
        self.requires_grad = a.requires_grad or b.requires_grad

        if a.requires_grad:
            def vjp_a(gradient: Tensor) -> Tensor:
                return Tensor(gradient.value.matmul(b.value.transpose()))
            self.parents.append(GraphNode(tensor=a, vjp=vjp_a))

        if b.requires_grad:
            def vjp_b(gradient: Tensor) -> Tensor:
                return Tensor(a.value.transpose().matmul(gradient.value))
            self.parents.append(GraphNode(tensor=b, vjp=vjp_b))


class Max(Tensor):

    def __init__(self, a: Tensor, axis: int) -> None:
        assert axis < 2, "Axis greater than 1"
        maxT, idxT = a.value.max(axis)
        super().__init__(maxT)
        self.requires_grad = a.requires_grad

        if a.requires_grad:
            def vjp(gradient: Tensor) -> Tensor:
                gradient, _ = _broadcast(gradient, a)
                grad = gradient.value * a.value.axial_mask(idxT, axis)
                return Tensor(grad)
            self.parents.append(GraphNode(tensor=a, vjp=vjp))


def Sigmoid(a: Tensor) -> Tensor:
    exp = a.exp()
    one = Tensor(1)
    if not a.onCPU():
        one.gpu()
    return exp / (one + exp)


class Relu(Tensor):

    def __init__(self, a: Tensor) -> None:
        super().__init__(a.value.relu())
        self.requires_grad = a.requires_grad

        if a.requires_grad:
            def vjp(gradient: Tensor) -> Tensor:
                return Tensor(a.value.relu_grad(gradient.value))
            self.parents.append(GraphNode(tensor=a, vjp=vjp))


def Mean(a: Tensor, axis: int = -1) -> Tensor:
    assert axis in [0, 1, -1], "Invalid axis"
    if axis == 0:
        length = Tensor(a.shape[0])
        a = a.sum(axis=axis)
        if not a.onCPU():
            length.gpu()
        return a / length
    elif axis == 1:
        length = Tensor(a.shape[1])
        a = a.sum(axis=axis)
        if not a.onCPU():
            length.gpu()
        return a / length
    else:
        length = Tensor(a.shape[0] * a.shape[1])
        a = a.sum(0).sum(1)
        if not a.onCPU():
            length.gpu()
        return a / length


def LogSoftmax(a: Tensor, axis: int):
    a_off: Tensor = a - a.max(axis)
    return a_off - (a_off.exp().sum(axis)).log()
