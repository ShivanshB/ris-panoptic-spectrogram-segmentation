import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

my_tensor = torch.tensor([[1, 2, 3],[4, 5, 6]], dtype=torch.float32, device=device, requires_grad=True)

print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.requires_grad)
print(my_tensor.shape)

# other common initialization techniques

x = torch.empty(size=(3,3)) # just fills w random stuff that's in memory
print(x)

x = torch.zeros((3,3)) # fills w zeros
print(x)

x = torch.rand((3,3)) # fills w uniform dist 0-1
print(x)

x = torch.ones((3,3)) # fills w/ all ones
print(x)

x = torch.eye(5,5) # identity matrix

x = torch.arange(start=0, end=5, step=1)
print(x)

x = torch.linspace(start=0.1, end=1, steps=10)
print(x)

x = torch.empty(size=(1,5)).normal_(mean=0, std=1)
print(x)

x = torch.empty(size=(1,5)).uniform_(0,1)
print(x)

x = torch.diag(torch.ones(3))
print(x)

# initializing and converting tensors to different types
tensor = torch.arange(4)
print(tensor)
print(tensor.dtype)

print(tensor.bool())
# only zero is false, so the first entry is true and the rest are false

print(tensor.short())
print(tensor.long())
print(tensor.half())
print(tensor.float())
print(tensor.double())

import numpy as np

np_array = np.zeros((5,5))
tensor = torch.from_numpy(np_array)
print(tensor)

np_array_back = tensor.numpy()
