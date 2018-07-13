import argparse
import os
import numpy as np
import math
import itertools

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs('images', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--lr', type=float, default=2e-4, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of second order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=32, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=10, help='dimension of the latent code')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=400, help='interval between image sampling')

opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def reparametrization(mu, logvar):
    std = logvar.mul(0.5).exp_()
    eps = std.data.new(std.size()).normal_()
    return eps.mul(std).add_(mu)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.mu = nn.Linear(512, opt.latent_dim)
        self.logvar = nn.Linear(512, opt.latent_dim)

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        x = self.model(img_flat)
        mu = self.mu(x)
        log_var = self.logvar(x)
        z = reparametrization(mu, log_var)
        return z


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img_flat = self.model(z)
        img = img_flat.view(img_flat.shape[0], *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        validity = self.model(z)
        return validity


adversarial_loss = torch.nn.BCELoss()
pixelwize_loss = torch.nn.L1Loss()

encoder = Encoder().to(device)
decoder = Decoder().to(device)
discriminator = Discriminator().to(device)

traindata = datasets.MNIST('../mnist', train=True,
                           transform=transforms.Compose([
                               transforms.Resize(opt.img_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,))
                           ]))

dataloader = DataLoader(traindata, batch_size=opt.batch_size, shuffle=True)

optimizer_G = torch.optim.Adam(itertools.chain(encoder.parameters(), decoder.parameters()),
                               lr=opt.lr, betas=(opt.b1, opt.b2))

optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


def sample_image(n_row, batches_done):
    z = torch.randn(n_row ** 2, opt.latent_dim).to(device)
    gen_imgs = decoder(z)
    save_image(gen_imgs.detach(), 'images/%d.png' % batches_done, nrow=n_row, normalize=True)


for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):
        valid = torch.Tensor(imgs.size(0), 1).fill_(1.0).to(device)
        fake = torch.Tensor(imgs.size(0), 1).fill_(0.0).to(device)

        real_imgs = imgs.to(device)

        optimizer_G.zero_grad()
        encoded_imgs = encoder(real_imgs)
        decoded_imgs = decoder(encoded_imgs)

        g_loss = 0.001 * adversarial_loss(discriminator(encoded_imgs),valid) + \
        0.999 * (decoded_imgs-real_imgs).pow(2).mean() # pixelwize_loss(decoded_imgs, real_imgs)

        g_loss.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()
        z = torch.randn(imgs.shape[0], opt.latent_dim).to(device)

        real_loss = adversarial_loss(discriminator(z), valid)
        fake_loss = adversarial_loss(discriminator(encoded_imgs.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)

        d_loss.backward()
        optimizer_D.step()

        if i % 400 == 0:
            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (
                epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item()))

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            sample_image(n_row=10, batches_done=batches_done)
