import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


class Sample:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self.data_min = -5
        self.data_max = 5

    def sample_true(self, N):
        x = torch.Tensor(N).normal_(self.mu, self.sigma).to('cuda:0')
        x, _ = x.sort()
        return x.view(-1, 1)

    def sample_noise(self, N):
        # sample from uniform distribution in the 3sigma interval
        x = torch.linspace(self.data_min, self.data_max, N) + torch.rand(N) * 0.01
        return x.view(-1, 1).to('cuda:0')
    

class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.l1 = torch.nn.Linear(1, 10)
        self.l1_prelu = torch.nn.PReLU()
        self.l2 = torch.nn.Linear(10, 10)
        self.l2_prelu = torch.nn.PReLU()
        self.l3 = torch.nn.Linear(10, 1)
        self.l3_sigmoid = torch.nn.Sigmoid()

    def forward(self, input):
        output = self.l1_prelu(self.l1(input))
        output = self.l2_prelu(self.l2(output))
        output = self.l3_sigmoid(self.l3(output))
        return output


class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.l1 = torch.nn.Linear(1, 10)
        self.l1_relu = torch.nn.ReLU()
        self.l2 = torch.nn.Linear(10, 10)
        self.l2_relu = torch.nn.ReLU()
        self.l3 = torch.nn.Linear(10, 1)

    def forward(self, x):
        x = self.l1_relu(self.l1(x))
        x = self.l2_relu(self.l2(x))
        x = self.l3(x)
        return x


def plot_fig(discriminate, generate, sample):
    xs = np.linspace(sample.data_min, sample.data_max, 1000)
    plt.plot(xs, norm.pdf(xs, loc=sample.mu, scale=sample.sigma))
    xs_tensor = torch.FloatTensor(xs).view(-1, 1).to('cuda:0')
    ds_tensor = discriminate(generate(xs_tensor))
    ds = ds_tensor.detach().cpu().numpy()
    plt.plot(xs, ds, label='decision boundary')
    zs_tensor = sample.sample_noise(int(1e3))
    gs_tensor = generate(zs_tensor)
    gs = gs_tensor.detach().cpu().numpy()
    plt.hist(gs, bins=10, density=True)
    axes = plt.gca()
    axes.set_ylim(0, 2)
    axes.set_xlim(-5, 5)

    # plt.show()


criterion = torch.nn.BCELoss()
discriminate = Discriminator().to('cuda:0')
generate = Generator().to('cuda:0')

optimizer_G = torch.optim.SGD(generate.parameters(), lr=0.1)
optimizer_D = torch.optim.SGD(discriminate.parameters(), lr=0.1)

# optimizer_G = torch.optim.Adam(generate.pa)

epochs = 300
mu = -2
sigma = 0.3
M = 200  # batch size
sampler = Sample(mu, sigma)


plot_fig(discriminate, generate, sampler)
plt.title('Before training')
plt.show()

log_d, log_g = np.zeros(epochs), np.zeros(epochs)
y_fake = torch.zeros(M, 1).to('cuda:0')
y_real = torch.ones(M, 1).to('cuda:0')


for epoch in range(epochs):

    real = sampler.sample_true(M)
    noise = sampler.sample_noise(M)
    fake = generate(noise)

    d_out_real = discriminate(real)
    d_out_fake = discriminate(fake)
    loss_D = 0.5 * criterion(d_out_fake, y_fake) + 0.5 * criterion(d_out_real, y_real)
    optimizer_D.zero_grad()
    loss_D.backward()
    optimizer_D.step()

    log_d[epoch] = loss_D.item()

    noise = sampler.sample_noise(M)
    fake = generate(noise)
    d_out_fake = discriminate(fake)
    loss_G = criterion(d_out_fake, y_real)
    optimizer_G.zero_grad()
    loss_G.backward()
    optimizer_G.step()
    log_g[epoch] = loss_G.item()

    if epoch % 10 == 0:
        print('Epoch: {} Discriminator loss {} \t Generator loss {}'.format(epoch, log_d[epoch], log_g[epoch]))


plt.plot(range(epochs), log_d, label='obj_d')
plt.plot(range(epochs), log_g, label='obj_g')
plt.legend()
plt.show()

plot_fig(discriminate, generate, sampler)
plt.title('After training')
plt.legend()
plt.show()
