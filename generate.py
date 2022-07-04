import torch


def reconstruct(vae, x, condition, device):
    
    x, condition = x[:1].to(device), condition[:1].unsqueeze(1).to(device)
    x_hat, _, _ = vae(x, condition)
    x = x.detach().cpu().numpy().squeeze().squeeze()
    x_hat = x_hat.detach().cpu().numpy().squeeze().squeeze()
        
    return x_hat


def generate(vae, condition, device):

    z_for_gen = torch.randn(1, vae.latent_dim).to(device)
    condition_for_gen = torch.LongTensor([[condition]]).to(device)
    gen_result = vae.decoder(z_for_gen, condition_for_gen)
    gen_result = gen_result.detach().cpu().numpy().reshape(28, 28)

    return gen_result
