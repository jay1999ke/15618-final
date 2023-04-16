import os
import sys
basepath = os.path.dirname(os.path.abspath(__file__))
tensorlibPath = "/build/"
sys.path.append(basepath+tensorlibPath)

import tensorlib
from autodiff.core.tensor import Tensor

