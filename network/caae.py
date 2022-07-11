import torch


class CAAEEncoder(torch.nn.Module):
    
    def __init__(self, cond_emb_dim, in_dim, hidden_dim, latent_dim, n_condition_labels, dropout_ratio=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.condition_emb = torch.nn.Embedding(n_condition_labels, cond_emb_dim)
        self.linear_in = torch.nn.Linear(in_dim+cond_emb_dim, hidden_dim)
        self.linear_hidden = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear_out = torch.nn.Linear(hidden_dim, latent_dim)
        self.dropout = torch.nn.Dropout(dropout_ratio)

    def forward(self, x, condition):
        x = x.flatten(1)
        condition = condition.unsqueeze(1)
        cond_emb = self.condition_emb(condition).flatten(1)
        x_c = torch.cat([x, cond_emb], -1)
        x_c = self.dropout(torch.nn.functional.elu(self.linear_in(x_c)))
        x_c = self.dropout(torch.tanh(self.linear_hidden(x_c)))
        z = self.linear_out(x_c)
        return z
    
    
class CAAEDecoder(torch.nn.Module):
    
    def __init__(self, cond_emb_dim, latent_dim, hidden_dim, out_dim, n_condition_labels, img_size, dropout_ratio=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.condition_emb = torch.nn.Embedding(n_condition_labels, cond_emb_dim)
        self.linear_in = torch.nn.Linear(latent_dim+cond_emb_dim, hidden_dim)
        self.linear_hidden = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear_out = torch.nn.Linear(hidden_dim, out_dim)
        self.dropout = torch.nn.Dropout(dropout_ratio)
        
    def forward(self, z, condition):
        condition = condition.unsqueeze(1)
        cond_emb = self.condition_emb(condition).flatten(1)
        z_c = torch.cat([z, cond_emb], -1)
        z_c = self.dropout(torch.tanh(self.linear_in(z_c)))
        z_c = self.dropout(torch.nn.functional.elu(self.linear_hidden(z_c)))
        x_hat = torch.sigmoid(self.linear_out(z_c))
        x_hat = x_hat.reshape((-1, 1, self.img_size, self.img_size))
        return x_hat


class ConditionalAdversarialAutoEncoder(torch.nn.Module):
    
    def __init__(self, cond_emb_dim, in_dim, latent_dim, hidden_dim, n_condition_labels, img_size, dropout_ratio=0.1):
        super().__init__()
        self.encoder = CAAEEncoder(cond_emb_dim, in_dim, hidden_dim, latent_dim, n_condition_labels, dropout_ratio)
        self.decoder = CAAEDecoder(cond_emb_dim, latent_dim, hidden_dim, in_dim, n_condition_labels, img_size, dropout_ratio)
        
    def forward(self, x, condition):
        z = self.encoder(x, condition)
        x_hat = self.decoder(z, condition)
        x_hat = x_hat.clamp(1e-8, 1-1e-8)
        return x_hat


class Discriminator(torch.nn.Module):
    
    def __init__(self, latent_dim, hidden_dim, out_dim, dropout_ratio=0.1):
        super().__init__()
        self.linear_in = torch.nn.Linear(latent_dim, hidden_dim)
        self.linear_out = torch.nn.Linear(hidden_dim, out_dim)
        self.dropout = torch.nn.Dropout(dropout_ratio)
        
    def forward(self, z):
        probs = self.dropout(torch.relu(self.linear_in(z)))
        probs = torch.sigmoid(self.linear_out(probs))
        return probs
