import torch


class VAEEncoder(torch.nn.Module):

    def __init__(self, in_dim, hidden_dim, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.linear_in = torch.nn.Linear(in_dim, hidden_dim)
        self.linear_hidden = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear_out = torch.nn.Linear(hidden_dim, latent_dim*2)

    def forward(self, x):
        x = x.flatten(1)
        x = torch.relu(self.linear_in(x))
        x = torch.relu(self.linear_hidden(x))
        x = self.linear_out(x)
        mu, sigma = x[:, :self.latent_dim], 1e-6+torch.nn.functional.softplus(x[:, self.latent_dim:])
        return mu, sigma


class VAEDecoder(torch.nn.Module):

    def __init__(self, latent_dim, hidden_dim, out_dim, img_size):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.linear_in = torch.nn.Linear(latent_dim, hidden_dim)
        self.linear_hidden = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear_out = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, z):
        z = torch.relu(self.linear_in(z))
        z = torch.relu(self.linear_hidden(z))
        x_hat = torch.sigmoid(self.linear_out(z))
        x_hat = x_hat.reshape((-1, 1, self.img_size, self.img_size))
        return x_hat


class VariationalAutoEncoder(torch.nn.Module):

    def __init__(self, in_dim, latent_dim, hidden_dim, img_size):
        super().__init__()
        self.encoder = VAEEncoder(in_dim, hidden_dim, latent_dim)
        self.decoder = VAEDecoder(latent_dim, hidden_dim, in_dim, img_size)
        
    def reparameterize(self, mu, sigma):
        epsilon = torch.randn_like(mu).to(mu.device)
        z = mu + sigma * epsilon
        return z

    def forward(self, x):
        mu, sigma = self.encoder(x)
        z = self.reparameterize(mu, sigma)
        x_hat = self.decoder(z)
        x_hat = x_hat.clamp(1e-8, 1-1e-8)
        return x_hat, mu, sigma
