import numpy as np

np.random.seed(0)

# Tensor dims
dim = (4,5)

# Initialize 3 Tensors
t1 = np.random.rand(dim[0], dim[1])
t2 = np.random.rand(dim[0], dim[1])
t3 = np.random.rand(dim[0], dim[1])

# Computational Graph
a = t1 + t2
b = a * t3
c = np.sum(b)

print('res: ', c)

gradc = 1.0
gradb = gradc * np.ones(dim)
grada = gradb * t3

gradt3 = gradb * a
gradt2 = grada.copy()
gradt1 = grada.copy()

print('grad 1: ', gradt1)
print('grad 2: ', gradt2)
print('grad 3: ', gradt3)