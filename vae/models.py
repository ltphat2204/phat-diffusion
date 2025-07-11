# vae/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=128, latent_dim=4):
        super().__init__()
        # 4 downsample blocks: 512→256→128→64→32
        self.conv1 = nn.Conv2d(in_channels, base_channels, 4, 2, 1)
        self.conv2 = nn.Conv2d(base_channels, base_channels*2, 4, 2, 1)
        self.conv3 = nn.Conv2d(base_channels*2, base_channels*4, 4, 2, 1)
        self.conv4 = nn.Conv2d(base_channels*4, base_channels*8, 4, 2, 1)

        self.flatten = nn.Flatten()
        hidden_dim = 32 * 32 * (base_channels*8)
        self.fc_mu     = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.flatten(x)
        mu     = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, out_channels=3, base_channels=128, latent_dim=4):
        super().__init__()
        hidden_dim = 32 * 32 * (base_channels*8)
        self.fc = nn.Linear(latent_dim, hidden_dim)

        self.deconv1 = nn.ConvTranspose2d(base_channels*8, base_channels*4, 4, 2, 1)
        self.deconv2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, 4, 2, 1)
        self.deconv3 = nn.ConvTranspose2d(base_channels*2, base_channels,   4, 2, 1)
        self.deconv4 = nn.ConvTranspose2d(base_channels,   out_channels,    4, 2, 1)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(z.size(0), -1, 32, 32)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = torch.tanh(self.deconv4(x))
        return x

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std
