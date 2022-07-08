import torch


class AEEncoder(torch.nn.Module):
    
    def __init__(self, in_dim, hidden_dim, latent_dim):
        super().__init__()
        self.linear_in = torch.nn.Linear(in_dim, hidden_dim)
        self.linear_hidden = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear_out = torch.nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        x = x.flatten(1)
        x = torch.relu(self.linear_in(x))
        x = torch.relu(self.linear_hidden(x))
        z = self.linear_out(x)
        return z
    
    
class AEDecoder(torch.nn.Module):
    
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


class AutoEncoder(torch.nn.Module):
    
    def __init__(self, in_dim, latent_dim, hidden_dim, img_size):
        super().__init__()
        self.encoder = AEEncoder(in_dim, hidden_dim, latent_dim)
        self.decoder = AEDecoder(latent_dim, hidden_dim, in_dim, img_size)
        
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        x_hat = x_hat.clamp(1e-8, 1-1e-8)
        return x_hat
