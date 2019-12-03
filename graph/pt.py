import torch
from torch.autograd import Variable

torch.manual_seed(0)

# Tensor dims
dim = (4,5)

# Initialize 3 Variables
t1 = Variable(torch.randn(dim[0], dim[1]), requires_grad=True)
t2 = Variable(torch.randn(dim[0], dim[1]), requires_grad=True)
t3 = Variable(torch.randn(dim[0], dim[1]), requires_grad=True)

# Computational Graph
a = t1 + t2
b = a * t3
c = torch.sum(b)

print('res: ', c)

# Gradient Computation
c.backward()

gradt1 = t1.grad.data
gradt2 = t1.grad.data
gradt3 = t1.grad.data

print('grad 1: ', gradt1)
print('grad 2: ', gradt2)
print('grad 3: ', gradt3)