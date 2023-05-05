# from torch import Tensor as pTensor
from autodiff.functional import *
import numpy as np
import time
import sys

numiter = 1000
warmup = 100

rows = [2, 4, 8, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
cols = [2, 4, 8, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
ops = [Add, Subtract, Multiply, Divide]    
lbl = ["Add", "Subtract", "Multiply", "Divide"]    

for j in range(len(rows)):

    row = rows[j]
    col = cols[j]

    np.random.seed(0)

    for kk in range(len(ops)):
        

        na = np.random.random((row,col))
        nb = np.random.random((row,col))

       
        a = Tensor(na)
        #a.requires_grad = True 
        b = Tensor(nb)
        #b.requires_grad = True

       
        op = ops[kk]
        a.gpu()
        b.gpu()
    

        for i in range(warmup):
            c = op(a,b)

        time_gpu = 0
        for i in range(numiter):
            t0 = time.time()
            c = op(a,b)
            t1 = time.time()
            time_gpu += t1 - t0


        a.cpu()
        b.cpu()
        
        for i in range(warmup):
            c = op(a,b)

        time_cpu = 0
        for i in range(numiter):
            t2 = time.time()
            c = op(a,b)
            t3 = time.time()
            time_cpu += t3 - t2

        
        print(lbl[kk], row, col, time_cpu, time_gpu, numiter, warmup, sys.argv[1])
