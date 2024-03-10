import torch
import numpy

#Commands for creating a tensors with the same dimensions

t1 = torch.randn((3,4))

t2 = torch.ones_like(t1)
t3 = torch.rand_like(t1)

# Other was to create a tensor

shape = (2,3,)
rand_tensor = torch.randn(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

#Attributes of a tensor

print("Attributes of Tensor 1")
print(f'Shape of tensor: {t1.shape}')
print(f'Datatype of tensor: {t1.dtype}')
print(f'Device tensor is stored on: {t1.device}')

# Move tensor onto the GPU

if torch.cuda.is_available():
    t1 = t1.to("cuda")

print(f'Device tensor is stored on: {t1.device}')

# Tensor Indexing

t4 = torch.ones((4,4))

print(f"First row: {t4[0, :]}")
print(f"First column: {t4[:, 0]}")
print(f"Last column: {t4[..., -1]}")

# Joining Tensors

t5 = torch.cat([t1,t1,t1], dim =1 )

# Arithmetic Operations

y1 = t1 @ t1.T
y2 = t1.matmul(t1.T)

print("\nShowing the equality of the two matrix multiplication methods")
print(y1==y2)
