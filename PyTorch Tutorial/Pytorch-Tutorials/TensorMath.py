import torch

x = torch.tensor([1, 2, 3])
y = torch.tensor([9, 8, 7])

# Addition
z1 = torch.empty(3)
torch.add(x, y, out=z1)
print(z1)

z2 = torch.add(x,y)

z = x + y

# Subtraction
z = x - y
print(z)

# Division
z = torch.true_divide(x ,y)
print(z)

# in place operations
t = torch.zeros(3)
t.add_(x) #operations with underscore after it are inplace

# exponentiation
z = x.pow(2)
z

z = x ** 2
z

# simple comparison
z = x > 0 # returns some sort of boolean array
z
z = x < 0
z

# Matrix Multiplication
x1 = torch.rand((2,5))
x2 = torch.rand((5,3))

x3 = torch.mm(x1, x2)
x3
x3 = x1.mm(x2)

# matrix exponentiation
matrix_exp = torch.rand(5,5)
matrix_exp = matrix_exp.matrix_power(5)
matrix_exp

# element wise Multiplication
z = x * y
z

# dot product
z = torch.dot(x,y)
z

# batch matrix Multiplication
batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand(batch, n, m)
tensor2 = torch.rand(batch, m, p)

out_bmm = torch.bmm(tensor1, tensor2)
out_bmm

# Broadcasting
x1 = torch.rand((5,5))
x2 = torch.rand((1,5))

z = x1 - x2
z #row subtracted across every row of x1

z = x1 ** x2
z

# other useful tensor operations
sum_x = torch.sum(x, dim=0)
sum_x

values, indices = torch.max(x, dim=0)
values
indices

abs_x = torch.abs(x)
abs_x

z = torch.argmax(x, dim=0)
z

z = torch.argmin(x, dim=0)
z

# arg operations just return indices of max values

mean_x = torch.mean(x.float(), dim=0)
mean_x

z = torch.eq(x,y)
z

torch.sort(y, dim=0, descending=False)

z = torch.clamp(x, min=0, max=10)
# any value < 0 set to 0, any val > 10 to 10

x = torch.tensor([1,0,1,1,1], dtype=torch.bool)
x

z = torch.any(x)
z

z = torch.all(x)
z
