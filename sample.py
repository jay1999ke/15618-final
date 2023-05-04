
# from torch import Tensor as pTensor
from autodiff import Tensor
import numpy as np


na = np.random.random((4,5))
nb = np.random.random((4,5))
nc = np.random.random((4,1))
nd = np.random.random((1,1))
ne = np.random.random((1,5))


# pa = pTensor(na)
# pa.requires_grad = True
a = Tensor(na)
a.requires_grad = True
a.gpu()

# pb = pTensor(nb)
# pb.requires_grad = True
b = Tensor(nb)
b.requires_grad = True
b.gpu()

# pc = pTensor(nc)
# pc.requires_grad = True
c = Tensor(nc)
c.requires_grad = True
c.gpu()

# pd = pTensor(nd)
# pd.requires_grad = True
d = Tensor(nd)
d.requires_grad = True
d.gpu()

# pe = pTensor(ne)
# pe.requires_grad = True
e = Tensor(ne)
e.requires_grad = True
e.gpu()

# pz = (pa + pb) * (pa*pb) + pc + pd + (pc*pd) + pe + (pc/pd) + (pc - pd) * (-pe)
# pz = pz.sum(0, keepdim=True).sum(1, keepdim=True)
z = (a + b) * (a*b) + c + d + (c*d) + e + (c/d) + (c - d) * (-e)
z = z.sum(0).sum(1)

# print(pz,z)
# pz.backward()

z.backward()

# ps = [pa,pb,pc,pd,pe]
s = [a,b,c,d,e]

# compare results: pytorch autograd vs autodiff
for q,w in zip(s, s):
    print(q.grad)
    print(w.grad)   
    print()