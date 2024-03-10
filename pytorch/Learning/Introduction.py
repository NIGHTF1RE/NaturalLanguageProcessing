import torch
import pandas as pd
import numpy as np

torch.manual_seed(1)

t1 = torch.tensor([1,2,3,4])
t2 = torch.tensor([[1,2,3],[4,5,6]])
t3 = torch.tensor(np.array([[1,2,3],[4,5,6]]))

print("Tensor 1")
print(t1)
print("\nTensor 2")
print(t2)
print("\nTensor 3")
print(t3)

t4 = torch.randn((2,3))
print("\nTensor 4")
print(t4)

print("\nSum of Tensor 3 and 4")
print(t3+t4)

print("\nConcatentation of Tensor 3 and 4")
print(torch.cat([t3,t4]))

print("\nRow aligned concatenation of Tensor 3 and 4")
print(torch.cat([t3,t4], dim=1))


#Reshaping Tensors

t5 = torch.randn(3,4)
print("\nTensor 5")
print(t5)
print("\nTensor 5 reshaped")
print(t5.view(2,6))

#Computation Graphs
t6 = torch.tensor([1., 2., 3], requires_grad=True)
t7 = torch.randn((1,3), requires_grad=True)

t8 = t6 + t7
print("\nShow grad_fn")
print(t8.grad_fn)

print("\nDemonstrate a lack of grad_fn passing when wrapped with no_grad")

print(t8.requires_grad)
print((t8 ** 2).requires_grad)

with torch.no_grad():
    print((t8 ** 2).requires_grad)