import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.2 * torch.rand(x.size())

class Net(nn.Module):
    def __init__(self, n_features, n_hidden_nuron, n_outputs):
        super(Net, self).__init__()
        self.hidden = nn.Linear(n_features, n_hidden_nuron)
        self.predict = nn.Linear(n_hidden_nuron, n_outputs)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


net = Net(1, 10, 1)

plt.ion()

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
loss_func = nn.MSELoss()

for t in range(100):
    prediction = net(x)
    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 5 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.show()

plt.ioff()
