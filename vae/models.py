import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Maps 512x512x3 → latent_dimx32x32
    """
    def __init__(self, in_channels=3, base_channels=128, latent_dim=4):
        super().__init__()
        # Downsampling: 512→256→128→64→32
        self.conv1 = nn.Conv2d(in_channels,           base_channels,   4, 2, 1)
        self.conv2 = nn.Conv2d(base_channels,         base_channels*2, 4, 2, 1)
        self.conv3 = nn.Conv2d(base_channels*2,       base_channels*4, 4, 2, 1)
        self.conv4 = nn.Conv2d(base_channels*4,       base_channels*8, 4, 2, 1)
        # Predict μ and log σ² as 1x1 convs, keeping 32x32 spatial dims
        self.conv_mu     = nn.Conv2d(base_channels*8, latent_dim,    1)
        self.conv_logvar = nn.Conv2d(base_channels*8, latent_dim,    1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))            # [B, base_channels*8, 32,32]
        mu     = self.conv_mu(x)             # [B, latent_dim, 32,32]
        logvar = self.conv_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    """
    Maps latent_dimx32x32 → 512x512x3
    """
    def __init__(self, out_channels=3, base_channels=128, latent_dim=4):
        super().__init__()
        self.conv_proj = nn.Conv2d(latent_dim, base_channels*8, 3, padding=1)
        # Upsampling: 32→64→128→256→512
        self.deconv1 = nn.ConvTranspose2d(base_channels*8, base_channels*4, 4, 2, 1)
        self.deconv2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, 4, 2, 1)
        self.deconv3 = nn.ConvTranspose2d(base_channels*2, base_channels,   4, 2, 1)
        self.deconv4 = nn.ConvTranspose2d(base_channels,   out_channels,    4, 2, 1)

    def forward(self, z):
        x = F.relu(self.conv_proj(z))        # [B, base_channels*8, 32,32]
        x = F.relu(self.deconv1(x))          # 32→64
        x = F.relu(self.deconv2(x))          # 64→128
        x = F.relu(self.deconv3(x))          # 128→256
        x = torch.tanh(self.deconv4(x))      # 256→512, output in [-1,1]
        return x

def reparameterize(mu, logvar):
    """
    z = μ + σ * ε with ε ~ N(0,1)
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std
