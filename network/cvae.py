import torch


class CVAEEncoder(torch.nn.Module):
    
    def __init__(self, cond_emb_dim, in_dim, hidden_dim, latent_dim, n_condition_labels):
        super().__init__()
        self.latent_dim = latent_dim
        self.condition_emb = torch.nn.Embedding(n_condition_labels, cond_emb_dim)
        self.linear_in = torch.nn.Linear(in_dim+cond_emb_dim, hidden_dim)
        self.linear_hidden = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear_out = torch.nn.Linear(hidden_dim, latent_dim*2)

    def forward(self, x, condition):
        x = x.flatten(1)
        condition = condition.unsqueeze(1)
        cond_emb = self.condition_emb(condition).flatten(1)
        x_c = torch.cat([x, cond_emb], -1)
        x_c = torch.relu(self.linear_in(x_c))
        x_c = torch.relu(self.linear_hidden(x_c))
        x_c = self.linear_out(x_c)
        mu, sigma = x_c[:, :self.latent_dim], 1e-6+torch.nn.functional.softplus(x_c[:, self.latent_dim:])
        return mu, sigma
    
    
class CVAEDecoder(torch.nn.Module):
    
    def __init__(self, cond_emb_dim, latent_dim, hidden_dim, out_dim, n_condition_labels, img_size):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.condition_emb = torch.nn.Embedding(n_condition_labels, cond_emb_dim)
        self.linear_in = torch.nn.Linear(latent_dim+cond_emb_dim, hidden_dim)
        self.linear_hidden = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear_out = torch.nn.Linear(hidden_dim, out_dim)
        
    def forward(self, z, condition):
        condition = condition.unsqueeze(1)
        cond_emb = self.condition_emb(condition).flatten(1)
        z_c = torch.cat([z, cond_emb], -1)
        z_c = torch.relu(self.linear_in(z_c))
        z_c = torch.relu(self.linear_hidden(z_c))
        x_hat = torch.sigmoid(self.linear_out(z_c))
        x_hat = x_hat.reshape((-1, 1, self.img_size, self.img_size))
        return x_hat


class ConditionalVariationalAutoEncoder(torch.nn.Module):
    
    def __init__(self, cond_emb_dim, in_dim, latent_dim, hidden_dim, n_condition_labels, img_size):
        super().__init__()
        self.encoder = CVAEEncoder(cond_emb_dim, in_dim, hidden_dim, latent_dim, n_condition_labels)
        self.decoder = CVAEDecoder(cond_emb_dim, latent_dim, hidden_dim, in_dim, n_condition_labels, img_size)

    def reparameterize(self, mu, sigma):
        epsilon = torch.randn_like(mu).to(mu.device)
        z = mu + sigma * epsilon
        return z
        
    def forward(self, x, condition):
        mu, sigma = self.encoder(x, condition)
        z = self.reparameterize(mu, sigma)
        x_hat = self.decoder(z, condition)
        x_hat = x_hat.clamp(1e-8, 1-1e-8)
        return x_hat, mu, sigma
