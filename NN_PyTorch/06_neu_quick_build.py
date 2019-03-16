import torch
import torch.nn.functional as F


n_data = torch.ones(100, 2)
x0 = torch.normal(2*n_data, 1)      # class0 x data (tensor), shape=(100, 2)
y0 = torch.zeros(100)               # class0 y data (tensor), shape=(100, 1)
x1 = torch.normal(-2*n_data, 1)     # class1 x data (tensor), shape=(100, 2)
y1 = torch.ones(100)                # class1 y data (tensor), shape=(100, 1)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # shape (200, 2) FloatTensor = 32-bit floating
y = torch.cat((y0, y1), ).type(torch.LongTensor)    # shape (200,) LongTensor = 64-bit integer


# build approach 1
class Net(torch.nn.Module):
    # define layers of neural network
    def __init__(self, n_feature, n_hidden_neuron, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden_neuron)
        self.predict = torch.nn.Linear(n_hidden_neuron, n_output)

    # build a neural network
    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.predict(x)
        return x


net1 = Net(2, 10, 2)

# quick build
net2 = torch.nn.Sequential(
    torch.nn.Linear(2, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 2)
)

print(net1)
print(net2)
