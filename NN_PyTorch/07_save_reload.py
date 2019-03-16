import torch
import torch.nn as nn
import matplotlib.pyplot as plt


x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1) #dim 1 to dim 2
y = x.pow(2) + 0.2*torch.rand(x.size())


def save():
    net1 = nn.Sequential(
        nn.Linear(1, 10),
        nn.ReLU(),
        nn.Linear(10, 1)
    )

    optimizer = torch.optim.SGD(net1.parameters(), lr=0.5)
    loss_func = nn.MSELoss()

    for t in range(100):
        prediction = net1(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # visualization of net1
    plt.figure(1, figsize=(10, 3))
    plt.subplot(131)
    plt.title('Net1')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)

    # save network in two ways
    torch.save(net1, 'net.pkl') # save the entire network
    torch.save(net1.state_dict(), 'net_params.pkl') # save the parameters inthe network


def reload_net():
    net2 = torch.load('net.pkl')
    prediction = net2(x)

    # visualization of net2
    plt.subplot(132)
    plt.title('Net2')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)


def reload_params():
    net3 = nn.Sequential(
        nn.Linear(1, 10),
        nn.ReLU(),
        nn.Linear(10, 1)
    )
    net3.load_state_dict(torch.load('net_params.pkl'))
    prediction = net3(x)

    # visualization of net3
    plt.subplot(133)
    plt.title('Net3')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    plt.show()


save() # save net1
reload_net() # reload the entire net1
reload_params() # only reload the parameters of net1
