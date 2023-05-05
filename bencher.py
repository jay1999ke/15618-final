# from torch import Tensor as pTensor
from autodiff.functional import *
import numpy as np
import time

numiter = 10
warmup = 100

rows = [2, 4, 8, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
cols = [2, 4, 8, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
ops = [Add]    

for j in range(len(rows)):

    r = rows[j]
    c = cols[j]

    na = np.random.random((r,c))
    nb = np.random.random((r,c))


    for i in range(len(ops)):
        
        a = Tensor(na)
        #a.requires_grad = True 
        b = Tensor(nb)
        #b.requires_grad = True

       
        op = ops[i]
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

        #print("Time on cpu:", time_cpu/numiter)
        #print("Time on gpu:", time_gpu/numiter)
        print("Speedup: ", time_cpu/time_gpu)
        #print("Gpu faster than cpu?", time_cpu > time_gpu)


