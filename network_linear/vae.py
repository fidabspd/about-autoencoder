import torch


class VAEEncoder(torch.nn.Module):
    
    def __init__(self, in_ch, hidden_ch, kernel_size, latent_dim, img_size):
        super().__init__()
        self.latent_dim = latent_dim
        self.conv0 = torch.nn.Conv2d(in_ch, hidden_ch, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv1 = torch.nn.Conv2d(hidden_ch, hidden_ch*2, kernel_size=kernel_size, padding=kernel_size//2)
        self.linear_out = torch.nn.Linear(hidden_ch*2*img_size*img_size, latent_dim*2)

    def forward(self, x):
        x = torch.relu(self.conv0(x))
        x = torch.relu(self.conv1(x)).flatten(1)
        x = self.linear_out(x)
        mu, sigma = x[:, :self.latent_dim], torch.exp(x[:, self.latent_dim:])
        return mu, sigma
    
    
class VAEDecoder(torch.nn.Module):
    
    def __init__(self, latent_dim, hidden_ch, kernel_size, out_ch, img_size):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_ch = hidden_ch
        self.img_size = img_size
        self.linear_in = torch.nn.Linear(latent_dim, hidden_ch//2*img_size*img_size)
        self.convt0 = torch.nn.ConvTranspose2d(hidden_ch//2, hidden_ch, kernel_size=kernel_size, padding=kernel_size//2)
        self.convt_out = torch.nn.ConvTranspose2d(hidden_ch, out_ch, kernel_size=kernel_size, padding=kernel_size//2)
        
    def forward(self, z):
        z = torch.relu(self.linear_in(z))
        z = z.reshape((-1, self.hidden_ch//2, self.img_size, self.img_size))
        z = torch.relu(self.convt0(z))
        x_hat = torch.sigmoid(self.convt_out(z))
        return x_hat


class VariationalAutoEncoder(torch.nn.Module):
    
    def __init__(self, in_ch, latent_dim, hidden_ch, kernel_size, img_size):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = VAEEncoder(in_ch, hidden_ch, kernel_size, latent_dim, img_size)
        self.decoder = VAEDecoder(latent_dim, hidden_ch, kernel_size, in_ch, img_size)

    def reparameterize(self, mu, sigma):
        epsilon = torch.randn(self.latent_dim).to(mu.device)
        z = mu + sigma * epsilon
        return z
        
    def forward(self, x):
        mu, sigma = self.encoder(x)
        z = self.reparameterize(mu, sigma)
        x_hat = self.decoder(z)
        return x_hat, mu, sigma