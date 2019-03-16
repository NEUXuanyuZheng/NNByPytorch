import torch
import numpy as np

# transform between numpy data and torch data
# np_data = np.arange(6).reshape((2, 3))
# torch_data = torch.from_numpy(np_data)
# tensor2array = torch_data.numpy()
#
# print(
#     '\nnumpy\n', np_data,
#     '\ntorch\n', torch_data,
#     '\ntensor2array\n', tensor2array
# )

# abs
# data = [-1, -2, 1, 2]
# tensor = torch.FloatTensor(data) # 32bit
# print(
#     '\nabs\n',
#     '\nnumpy\n', np.abs(data),
#     '\ntorch\n', torch.abs(tensor)
# )

data = [[1,2], [3,4]]
data = np.array(data)
tensor = torch.FloatTensor(data) # 32bit
print(
    # also you can use data.dot(data) to do matrix multiply in numpy
    '\nnumpy\n', np.matmul(data, data), 
    '\ntorch\n', torch.mm(tensor, tensor)
)