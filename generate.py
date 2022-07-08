import torch


def reconstruct(model, x, condition=None, output_only_x=False):
    
    if condition is None:
        x_hat = model(x)
    else:
        x_hat = model(x, condition)
    if not output_only_x:
        x_hat = x_hat[0]
    x = x.detach().cpu().numpy().squeeze().squeeze()
    x_hat = x_hat.detach().cpu().numpy().squeeze().squeeze()
        
    return x, x_hat


def generate(generator, image_size, condition=None):

    device = 'cuda' if next(generator.parameters()).is_cuda else 'cpu'

    z = torch.randn(1, generator.latent_dim).to(device)
    if condition is None:
        gen_result = generator(z)
    else:
        condition = torch.LongTensor([condition]).to(device)
        gen_result = generator(z, condition)
    gen_result = gen_result.detach().cpu().numpy().reshape(image_size, image_size)

    return gen_result
