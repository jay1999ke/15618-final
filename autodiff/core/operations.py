from autodiff import tensorlib


class Operations:
    add = "add"
    mul = "mul"


CPU = {
    Operations.add: tensorlib.cpu_add,
    Operations.mul: tensorlib.cpu_mul,
}

GPU = {
    Operations.add: tensorlib.gpu_add,
    Operations.mul: tensorlib.gpu_mul,
}
