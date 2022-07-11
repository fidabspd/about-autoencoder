import torch


class AAEEncoder(torch.nn.Module):
    
    def __init__(self, in_dim, hidden_dim, latent_dim, dropout_ratio=0.1):
        super().__init__()
        self.linear_in = torch.nn.Linear(in_dim, hidden_dim)
        self.linear_hidden = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear_out = torch.nn.Linear(hidden_dim, latent_dim)
        self.dropout = torch.nn.Dropout(dropout_ratio)

    def forward(self, x):
        x = x.flatten(1)
        x = self.dropout(torch.nn.functional.elu(self.linear_in(x)))
        x = self.dropout(torch.tanh(self.linear_hidden(x)))
        z = self.linear_out(x)
        return z
    
    
class AAEDecoder(torch.nn.Module):
    
    def __init__(self, latent_dim, hidden_dim, out_dim, img_size, dropout_ratio=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.linear_in = torch.nn.Linear(latent_dim, hidden_dim)
        self.linear_hidden = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear_out = torch.nn.Linear(hidden_dim, out_dim)
        self.dropout = torch.nn.Dropout(dropout_ratio)
        
    def forward(self, z):
        z = self.dropout(torch.tanh(self.linear_in(z)))
        z = self.dropout(torch.nn.functional.elu(self.linear_hidden(z)))
        x_hat = torch.sigmoid(self.linear_out(z))
        x_hat = x_hat.reshape((-1, 1, self.img_size, self.img_size))
        return x_hat


class AdversarialAutoEncoder(torch.nn.Module):
    
    def __init__(self, in_dim, latent_dim, hidden_dim, img_size, dropout_ratio=0.1):
        super().__init__()
        self.encoder = AAEEncoder(in_dim, hidden_dim, latent_dim, dropout_ratio)
        self.decoder = AAEDecoder(latent_dim, hidden_dim, in_dim, img_size, dropout_ratio)
        
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        x_hat = x_hat.clamp(1e-8, 1-1e-8)
        return x_hat


class Discriminator(torch.nn.Module):
    
    def __init__(self, latent_dim, hidden_dim, out_dim, dropout_ratio=0.1):
        super().__init__()
        self.linear_in = torch.nn.Linear(latent_dim, hidden_dim)
        self.linear_out = torch.nn.Linear(hidden_dim, out_dim)
        self.dropout = torch.nn.Dropout(dropout_ratio)
        
    def forward(self, z):
        z = self.dropout(torch.relu(self.linear_in(z)))
        probs = torch.sigmoid(self.linear_out(z))
        return probs
