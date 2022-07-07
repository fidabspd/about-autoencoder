import torch


class LogLikelihood(torch.nn.Module):
    
    def __init__(self, scale=1.):
        super().__init__()
        self.scale = scale
        
    def forward(self, x, x_hat):
        log_likelihood = torch.mean(x * torch.log(x_hat) + (1 - x) * torch.log(1-x_hat))
        return log_likelihood * self.scale


class KLDivergence(torch.nn.Module):

    def __init__(self, scale=1.):
        super().__init__()
        self.scale = scale

    def forward(self, mu, sigma):
        kl_divergence = 0.5 * torch.mean(torch.square(mu) + torch.square(sigma) - torch.log(torch.square(sigma)) - 1)
        return kl_divergence * self.scale


class ELBO(torch.nn.Module):
    
    def __init__(self, scale=1.):
        super().__init__()
        self.log_likelihood = LogLikelihood(scale)
        self.kl_divergence = KLDivergence(scale)
        
    def forward(self, x, x_hat, mu, sigma):
        _log_likelihood = self.log_likelihood(x, x_hat)
        _kl_divergence = self.kl_divergence(mu, sigma)
        elbo = _log_likelihood - _kl_divergence
        return elbo, _log_likelihood, _kl_divergence


class DiscriminatorLoss(torch.nn.Module):
    
    def __init__(self, scale=1.):
        super().__init__()
        self.scale = scale
        self.bce_loss = torch.nn.BCELoss()
        
    def forward(self, real_disc_probs, fake_disc_probs):
        check = torch.sum(torch.isnan(torch.log(real_disc_probs))).item() + \
            torch.sum(torch.isnan(torch.log(fake_disc_probs))).item()
        if check > 0:
            print(real_disc_probs)
            print(fake_disc_probs)
            raise ValueError("There is nan.")
        try:
            real_loss = self.bce_loss(real_disc_probs, torch.ones_like(real_disc_probs))
            fake_loss = self.bce_loss(fake_disc_probs, torch.zeros_like(fake_disc_probs))
        except:
            print(real_disc_probs, fake_disc_probs)
            raise ValueError("Something happend in output value of bce ...")
        discriminator_loss = (real_loss + fake_loss) * self.scale
        return discriminator_loss


class GeneratorLoss(torch.nn.Module):

    def __init__(self, scale=1.):
        super().__init__()
        self.scale = scale
        self.bce_loss = torch.nn.BCELoss()
        
    def forward(self, fake_disc_probs):
        generator_loss = self.bce_loss(fake_disc_probs, torch.ones_like(fake_disc_probs))
        generator_loss = generator_loss * self.scale
        return generator_loss
