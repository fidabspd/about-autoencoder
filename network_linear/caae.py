import torch


class CAAEEncoder(torch.nn.Module):
    
    def __init__(self, cond_emb_dim, in_ch, hidden_ch, kernel_size, latent_dim, n_condition_labels, img_size):
        super().__init__()
        self.latent_dim = latent_dim
        self.condition_emb = torch.nn.Embedding(n_condition_labels, cond_emb_dim)
        self.conv0 = torch.nn.Conv2d(in_ch, hidden_ch, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv1 = torch.nn.Conv2d(hidden_ch, hidden_ch*2, kernel_size=kernel_size, padding=kernel_size//2)
        self.linear_out = torch.nn.Linear(hidden_ch*2*img_size*img_size+cond_emb_dim, latent_dim)

    def forward(self, x, condition):
        cond_emb = self.condition_emb(condition).flatten(1)
        x = torch.relu(self.conv0(x))
        x = torch.relu(self.conv1(x)).flatten(1)
        x_c = torch.cat([x, cond_emb], 1)
        z = self.linear_out(x_c)
        return z
    
    
class CAAEDecoder(torch.nn.Module):
    
    def __init__(self, cond_emb_dim, latent_dim, hidden_ch, kernel_size, out_ch, n_condition_labels, img_size):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_ch = hidden_ch
        self.img_size = img_size
        self.condition_emb = torch.nn.Embedding(n_condition_labels, cond_emb_dim)
        self.linear_in = torch.nn.Linear(latent_dim+cond_emb_dim, hidden_ch//2*img_size*img_size)
        self.convt0 = torch.nn.ConvTranspose2d(hidden_ch//2, hidden_ch, kernel_size=kernel_size, padding=kernel_size//2)
        self.convt_out = torch.nn.ConvTranspose2d(hidden_ch, out_ch, kernel_size=kernel_size, padding=kernel_size//2)
        
    def forward(self, z, condition):
        cond_emb = self.condition_emb(condition).flatten(1)
        z_c = torch.cat([z, cond_emb], 1)
        z_c = torch.relu(self.linear_in(z_c))
        z_c = z_c.reshape((-1, self.hidden_ch//2, self.img_size, self.img_size))
        z_c = torch.relu(self.convt0(z_c))
        x_hat = torch.sigmoid(self.convt_out(z_c))
        return x_hat


class ConditionalAdversarialAutoEncoder(torch.nn.Module):
    
    def __init__(self, cond_emb_dim, in_ch, latent_dim, hidden_ch, kernel_size, n_condition_labels, img_size):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = CAAEEncoder(cond_emb_dim, in_ch, hidden_ch, kernel_size, latent_dim, n_condition_labels, img_size)
        self.decoder = CAAEDecoder(cond_emb_dim, latent_dim, hidden_ch, kernel_size, in_ch, n_condition_labels, img_size)
        
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        x_hat = x_hat.clamp(1e-8, 1-1e-8)
        return x_hat


class Discriminator(torch.nn.Module):
    
    def __init__(self, latent_dim, hidden_dim, out_dim):
        super().__init__()
        self.linear_in = torch.nn.Linear(latent_dim, hidden_dim)
        self.linear_out = torch.nn.Linear(hidden_dim, out_dim)
        
    def forward(self, z):
        probs = torch.relu(self.linear_in(z))
        probs = torch.sigmoid(self.linear_out(probs))
        return probs
