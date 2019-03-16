import torch
from torch.autograd import Variable


tensor = torch.FloatTensor([[1, 2], [3, 4]])
# requires_grad: whether to calculate gradient
variable = Variable(tensor, requires_grad=True)

# print(tensor)
# print(variable)

t_out = torch.mean(tensor*tensor)
v_out = torch.mean(variable*variable)

# print(t_out)
# print(v_out)

# v_out.backward()
# # v_out = 1/4*sum(var*var)
# # d(v_out)/d(var) = 1/4*2*var = var/2
# print(variable.grad)

print(variable)

print(variable.data)

print(variable.data.numpy())

