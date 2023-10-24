import torch
import numpy as np

# Tensors are a specialized data structure
# They are similar to arrays and matrices
# In pyTorch they are used to encode inputs and outputs of a model
# as well as its parameters

# tensor directly from data
data = [[ 1, 2 ], [ 3, 4 ]]
x_data = torch.tensor(data)

#tensor from numpy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# Tensor from another tensor
# the new tensor retains properties
# like shape and datatype unless otherwise stated
x_ones =torch.ones_like(x_data) # retains properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides datatype to be a float
print(f"Random Tensor: \n {x_rand} \n")

# you can pass a tuple as the shape of the tensor
shape = (2, 3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor} \n")

# Tensor atributes describe their shape, type, and device they are stored on.
tensor = torch.rand(3,4)
print(f"Shape of Tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

### we can now perform various operations on tensors ###

# Standard numpy like indexing and slicing.
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First Column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)

# Can use torch cat to concatenate a sequence of tensors
# along a given dimension.
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
# ``tensor.T`` returns the transpose of a tensor
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)


# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

# if you have a single element tensor you can convert it to
# a python numerical value using item()
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

# in place operations store the result into the original tensor
# they are denoted by a _ suffix
print(f"{tensor} \n")
tensor.add_(5)
print(tensor)

# Tensors on the CPU and NumPy arrays can share their underlying memory locations, and changing one will change the other.
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")
# a change to the tensor changes the numpy array
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# a change in the numpy array can effect the tensor
n = np.ones(5)
t = torch.from_numpy(n)
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")


