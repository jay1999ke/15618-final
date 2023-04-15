import blas
import numpy as np

arr2 = np.linspace(1.0,9,9).astype(np.float32).reshape(3,3)
arr1 = np.linspace(1.0,9,9).astype(np.float32).reshape(3,3)

print("1  ", arr1)
print("2  ", arr2)
print("cpu", blas.cpu_add(arr1, arr2))
print("gpu", blas.gpu_add(arr1, arr2))

class Tensor(object):

    def __init__(self) -> None:
        self.value: np.ndarray = None