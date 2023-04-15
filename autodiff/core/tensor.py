import blas
import numpy as np

arr2 = np.linspace(1.0,9,9).astype(np.float32).reshape(3,3)
arr1 = np.linspace(1.0,9,9).astype(np.float32).reshape(3,3)

arr3 = blas.cpu_add(arr1,arr2)

print(arr3)

class Tensor(object):

    def __init__(self) -> None:
        self.value: np.ndarray = None