import torch

x = torch.arange(9)
x_3x3 = x.view(3,3)
x_3x3
x_3x3 = x.reshape(3,3)
x_3x3
y = x_3x3.t()
y
x1 = torch.rand((1,5))
x2 = torch.rand((1,5))

print(torch.cat((x1,x2), dim=0))
