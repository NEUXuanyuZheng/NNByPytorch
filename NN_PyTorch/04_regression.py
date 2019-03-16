import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt


x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1) #dim 1 to dim 2
y = x.pow(2) + 0.2*torch.rand(x.size())

x, y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()


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


net = Net(1, 10, 1)
print(net)

plt.ion()

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
loss_func = torch.nn.MSELoss()

for t in range(100):
    prediction = net(x)

    loss = loss_func(prediction, y)

    optimizer.zero_grad() # set all gradient 0
    loss.backward()
    optimizer.step()
    if t % 5 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()