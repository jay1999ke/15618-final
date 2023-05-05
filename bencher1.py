# from torch import Tensor as pTensor
from autodiff.functional import *
import numpy as np
import time
import sys

numiter = 100
warmup = 10

rows = [2, 4, 8, 64, 128, 256, 512, 1024]
cols = [2, 4, 8, 64, 128, 256, 512, 1024]
ops = [Add, Subtract, Multiply, Divide, MatMul]    
lbl = ["Add", "Subtract", "Multiply", "Divide", "MatMul"]    

for j in range(len(rows)):

    row = rows[j]
    col = cols[j]

    np.random.seed(0)

    for kk in range(len(ops)):
        

        na = np.random.random((row,col))
        nb = np.random.random((row,col))
        no = np.ones((row,col))
       
        a = Tensor(na)
        a.requires_grad = True 
        b = Tensor(nb)
        b.requires_grad = True
        ones = Tensor(no)
       
        op = ops[kk]
        a.gpu()
        b.gpu()
        ones.gpu()    
        
        for i in range(warmup):
            c = op(a,b)

        time_gpu_forward = 0
        time_gpu_back = 0
        for i in range(numiter):
            t0 = time.time()
            c = op(a,b)
            t1 = time.time()
            c.backward(ones)
            t2 = time.time()
            time_gpu_forward += t1 - t0
            time_gpu_back += t2 - t1

        a.cpu()
        b.cpu()
        ones.cpu()

        for i in range(warmup):
            c = op(a,b)

        time_cpu_forward = 0
        time_cpu_back = 0
        for i in range(numiter):
            t3 = time.time()
            c = op(a,b)
            t4 = time.time()
            c.backward(ones)
            t5 = time.time()
            time_cpu_forward += t4 - t3
            time_cpu_back += t5 - t4
        
        print(lbl[kk], row, col, time_cpu_forward, time_cpu_back, time_gpu_forward, time_gpu_back, numiter, warmup, sys.argv[1])
