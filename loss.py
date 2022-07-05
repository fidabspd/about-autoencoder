import torch


class LogLikelihood(torch.nn.Module):
    
    def __init__(self, scale=1.):
        super().__init__()
        self.scale = scale
        
    def forward(self, x, x_hat):
        log_likelihood = torch.mean(x * torch.log(x_hat) + (1 - x) * torch.log(1 - x_hat))
        return log_likelihood * self.scale


class KLDivergence(torch.nn.Module):

    def __init__(self, scale=1.):
        super().__init__()
        self.scale = scale

    def forward(self, mu, sigma):
        kl_divergence = 0.5 * torch.mean(torch.square(mu) + torch.square(sigma) - torch.log(1e-8 + torch.square(sigma)) - 1)
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
