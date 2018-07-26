import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.utils import save_image
import torchvision.transforms as transforms
import torch.nn as nn
import os

if not os.path.exists('./images'):
    os.mkdir('./images')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(100, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, 28*28),
            nn.Tanh() # Tanh activation, Leaky ReLU, batch normalization, fight against mode collapse
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 1, 28, 28)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity


adversarial_loss = torch.nn.BCELoss().to(device)

generator = Generator().to(device)
discriminator = Discriminator().to(device)

train_data = datasets.MNIST('../mnist', train=True, download=False, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
]))

dataloader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))


for epoch in range(200):
    for i, (imgs, _) in enumerate(dataloader):
        valid = torch.Tensor(imgs.size(0), 1).fill_(1.0).to(device)
        fake = torch.Tensor(imgs.size(0), 1).fill_(0.0).to(device)

        real_imgs = imgs.to(device)
        z = torch.randn((imgs.shape[0], 100)).to(device)
        gen_imgs = generator(z)

        pred_gen = discriminator(gen_imgs)
        pred_real = discriminator(real_imgs)

        optimizer_D.zero_grad()
        real_loss = adversarial_loss(pred_real, valid)
        fake_loss = adversarial_loss(pred_gen, fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward(retain_graph=True)
        optimizer_D.step()

        g_loss = adversarial_loss(pred_gen, valid)
        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

        print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"%(epoch, 200, i, len(dataloader), d_loss.item(), g_loss.item()))
        batches_done = epoch*len(dataloader)+i
        if batches_done%400==0:
            save_image(gen_imgs.cpu()[:25], 'images/%d.png'%batches_done, nrow=5, normalize=True)


