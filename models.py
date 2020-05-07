import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Interpolate(nn.Module):
    def __init__(self, scale_factor=2):
        super(Interpolate, self).__init__()
        self.interp = F.interpolate
        self.scale_factor = scale_factor
        
    def forward(self, x):
        x = self.interp(x, scale_factor = self.scale_factor)
        return x


mnist_lenet_encoder = nn.Sequential(
    nn.Conv2d(1, 8, 5, padding=2),
    nn.BatchNorm2d(8, affine=False),
    nn.LeakyReLU(negative_slope=0.1),
    nn.MaxPool2d(2,2),
    nn.Conv2d(8, 4, 5, padding=2),
    nn.BatchNorm2d(4, affine=False),
    nn.LeakyReLU(negative_slope=0.1),
    nn.MaxPool2d(2,2),
)

mnist_lenet_decoder = nn.Sequential(
    nn.LeakyReLU(negative_slope=0.1),
    Interpolate(2),
    nn.ConvTranspose2d(2, 4, 5, padding=2),
    nn.BatchNorm2d(4, affine=False),
    nn.LeakyReLU(negative_slope=0.1),
    Interpolate(2),
    nn.ConvTranspose2d(4, 8, 5, padding=3),
    nn.BatchNorm2d(8, affine=False),
    nn.LeakyReLU(negative_slope=0.1),
    Interpolate(2),
    nn.ConvTranspose2d(8, 1, 5, padding=2),
    nn.Sigmoid()
)

mnist_encoding_dim = 4 * 7 * 7

cifar10_lenet_encoder = nn.Sequential(
    nn.Conv2d(3, 32, 5, padding=2),
    nn.BatchNorm2d(32, affine=False),
    nn.LeakyReLU(negative_slope=0.1),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(32, 64, 5, padding=2),
    nn.BatchNorm2d(64, affine=False),
    nn.LeakyReLU(negative_slope=0.1),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(64, 128, 5, padding=2),
    nn.BatchNorm2d(128, affine=False),
    nn.LeakyReLU(negative_slope=0.1),
    nn.MaxPool2d(2, 2),
)

cifar10_lenet_decoder = nn.Sequential(
    nn.LeakyReLU(negative_slope=0.1),
    nn.ConvTranspose2d(int(128 / (4 * 4)), 128, 5, padding=2),
    nn.BatchNorm2d(128, affine=False),
    nn.LeakyReLU(negative_slope=0.1),
    Interpolate(2),
    nn.ConvTranspose2d(128, 64, 5, padding=2),
    nn.BatchNorm2d(64, affine=False),
    nn.LeakyReLU(negative_slope=0.1),
    Interpolate(2),
    nn.ConvTranspose2d(64, 32, 5, padding=2),
    nn.BatchNorm2d(32, affine=False),
    nn.LeakyReLU(negative_slope=0.1),
    Interpolate(2),
    nn.ConvTranspose2d(32, 3, 5, padding=2),
    nn.Sigmoid()
)

cifar10_encoding_dim = 128 * 4 * 4

class PGN(nn.Module):
    """
    Prior Generating Network
    """
    def __init__(self, rep_dim, encoding_dim, dropoutp, features, dtype):
        super().__init__()
        self.dtype = dtype
        self.dropoutp = dropoutp
        self.features = features
        self.fc = nn.Linear(encoding_dim, encoding_dim)
        self.dense_mu = nn.Linear(encoding_dim, rep_dim)
        self.dense_logvar = nn.Linear(encoding_dim, rep_dim)
        self.mcdropout = True
        
    def set_mcdropout(self, mcdropout):
        self.mcdropout = mcdropout

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        h = self.fc(x)
        mu = self.dense_mu(F.dropout(h, p = self.dropoutp, training = self.mcdropout))
        logvar = self.dense_logvar(h)
        return mu, logvar
    

class Autoencoder(nn.Module):
    def __init__(self, rep_dim, encoding_dim, encoder, decoder):
        super().__init__()

        self.rep_dim = rep_dim
        self.encoding_dim = encoding_dim
        self.encoder = encoder
        self.fc1 = nn.Linear(encoding_dim, self.rep_dim)
        self.decoder = decoder
        
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = x.view(x.size(0), int(self.rep_dim / 16), 4, 4)
        x = self.decoder(x)
        return x
        
class VAE(nn.Module):
    def __init__(self, rep_dim, encoding_dim, encoder, decoder, L=10):
        super().__init__()
        self.L = L  # the number of reparameterization
        self.rep_dim = rep_dim
        self.encoder = encoder
        self.decoder = decoder

        self.fc1_mu = nn.Linear(encoding_dim, self.rep_dim)
        self.fc1_logvar = nn.Linear(encoding_dim, self.rep_dim)

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc1_mu(x)
        logvar = self.fc1_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        z = z.view(z.size(0), int(self.rep_dim / 16), 4, 4)
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        recon_list = []
        for l in range(self.L):
            z = self.reparameterize(mu, logvar)
            recon_list.append(self.decode(z))
        return recon_list, mu, logvar
        

class AAE_Encoder(nn.Module):
    def __init__(self, rep_dim, encoding_dim, encoder):
        super().__init__()
        self.rep_dim = rep_dim
        self.encoder = encoder
        self.fc1 = nn.Linear(encoding_dim, rep_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.fc1(x)


class AAE_Decoder(nn.Module):
    def __init__(self, rep_dim, decoder):
        super().__init__()
        self.rep_dim = rep_dim
        self.decoder = decoder
    
    def forward(self, x):
        x = x.view(x.size(0), int(self.rep_dim / 16), 4, 4)
        return self.decoder(x)


class AAE_Discriminator(nn.Module):
    def __init__(self, rep_dim):
        super().__init__()
        self.rep_dim = rep_dim
        self.model = nn.Sequential(
            nn.Linear(rep_dim, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        validity = self.model(z)
        return validity

class GPND_Discriminator(nn.Module):
    def __init__(self, encoding_dim, encoder):
        super().__init__()
        self.encoder = encoder
        self.fc1 = nn.Linear(encoding_dim, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return torch.sigmoid(x)


class ANOGAN_Discriminator(nn.Module):
    def __init__(self, encoding_dim, encoder):
        super().__init__()
        self.encoder = encoder
        self.fc1 = nn.Linear(encoding_dim, 1)

    def forward(self, x):
        x = self.encoder(x)
        out = x.view(x.size(0), -1)
        out2 = self.fc1(out)
        return torch.sigmoid(out2), out
        


def get_pgn_encoder(datatype, p=0.2):
    if datatype.lower() in ['mnist','fmnist']:
        return PGN(rep_dim=32, encoding_dim=mnist_encoding_dim, dropoutp=p, features=mnist_lenet_encoder, dtype=datatype)
    elif datatype.lower() in ['cifar10']:
        return PGN(rep_dim=128, encoding_dim=cifar10_encoding_dim, dropoutp=p, features=cifar10_lenet_encoder, dtype=datatype)

def get_dsvdd(datatype):
    if datatype.lower() in ['mnist','fmnist']:
        return DSVDD(rep_dim=32, encoding_dim=mnist_encoding_dim, features=mnist_lenet_encoder)
    elif datatype.lower() in ['cifar10']:
        return DSVDD(rep_dim=128, encoding_dim=cifar10_encoding_dim, features=cifar10_lenet_encoder)


def get_ae(datatype):
    if datatype.lower() in ['mnist','fmnist']:
        return Autoencoder(rep_dim = 32, encoding_dim = mnist_encoding_dim, encoder = mnist_lenet_encoder, decoder = mnist_lenet_decoder)
    elif datatype.lower() in ['cifar10']:
        return Autoencoder(rep_dim = 128, encoding_dim = cifar10_encoding_dim, encoder = cifar10_lenet_encoder, decoder = cifar10_lenet_decoder)

def get_vae(datatype, L=10):
    if datatype.lower() in ['mnist', 'fmnist']:
        return VAE(rep_dim = 32, encoding_dim = mnist_encoding_dim, encoder = mnist_lenet_encoder, decoder = mnist_lenet_decoder, L=L)
    elif datatype.lower() in ['cifar10']:
        return VAE(rep_dim = 128, encoding_dim = cifar10_encoding_dim, encoder = cifar10_lenet_encoder, decoder = cifar10_lenet_decoder, L=L)


def get_aae(datatype):
    if datatype.lower() in ['mnist', 'fmnist']:
        rep_dim = 32
        encoding_dim = mnist_encoding_dim
        encoder = mnist_lenet_encoder
        decoder = mnist_lenet_decoder
    elif datatype.lower() in ['cifar10']:
        rep_dim = 128
        encoding_dim = cifar10_encoding_dim
        encoder = cifar10_lenet_encoder
        decoder = cifar10_lenet_decoder

    aae_encoder = AAE_Encoder(rep_dim=rep_dim, encoding_dim=encoding_dim, encoder=encoder)
    aae_decoder = AAE_Decoder(rep_dim=rep_dim, decoder=decoder)
    aae_discriminator = AAE_Discriminator(rep_dim=rep_dim)
    return aae_encoder, aae_decoder, aae_discriminator

def get_gpnd(datatype):
    if datatype.lower() in ['mnist', 'fmnist']:
        rep_dim = 32
        encoding_dim = mnist_encoding_dim
        encoder = mnist_lenet_encoder
        decoder = mnist_lenet_decoder
    elif datatype.lower() in ['cifar10']:
        rep_dim = 128
        encoding_dim = cifar10_encoding_dim
        encoder = cifar10_lenet_encoder
        decoder = cifar10_lenet_decoder

    gpnd_encoder = AAE_Encoder(rep_dim=rep_dim, encoding_dim=encoding_dim, encoder=encoder)
    gpnd_decoder = AAE_Decoder(rep_dim=rep_dim, decoder=decoder)
    gpnd_discriminator = GPND_Discriminator(encoding_dim=encoding_dim, encoder=encoder)
    gpnd_z_discriminator = AAE_Discriminator(rep_dim=rep_dim)
    return gpnd_encoder, gpnd_decoder, gpnd_discriminator, gpnd_z_discriminator


class DSVDD(nn.Module):
    def __init__(self, datatype):
        super().__init__()
        self.datatype = datatype.lower()
        if datatype.lower() in ['mnist','fmnist']:
            self.features = mnist_lenet_encoder
            self.fc = nn.Linear(4 * 7 * 7, 32, bias = False)
            self.rep_dim = 32
        elif datatype.lower() in ['cifar10']:
            self.features = cifar10_lenet_encoder
            self.fc = nn.Linear(128 * 4 * 4, 128, bias = False)
            self.rep_dim = 128

    def forward(self, x):
        x = self.features(x)
        out = x.view(x.size(0), -1)
    
        h = self.fc(out)

        return h
