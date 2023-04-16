from autodiff import tensorlib

class Operations:
    add = "add"

CPU = {
    "add": tensorlib.cpu_add
}

GPU = {
    "add": tensorlib.gpu_add
}
