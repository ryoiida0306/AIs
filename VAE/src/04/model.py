import torch
import torch.nn as nn
import numpy as np

class Model(nn.Module) :

    def __init__(self, input_output_dim, hidden_dim, z_dim) :
        super(Model, self).__init__()
        self.flatten = nn.Flatten()
        self.encoder = nn.Sequential(
            nn.Linear(input_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim*2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, input_output_dim),
            nn.Sigmoid()

        )
    
    def reparameterization_trick(self, mu, log_var) :
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x) :
        x = self.flatten(x)
        mu, log_var = self.encoder(x).chunk(2, dim=1)
        log_var = torch.clamp(log_var, min=-1e+6)
        z = self.reparameterization_trick(mu, log_var)
        z = self.decoder(z)
        log_var = torch.clamp(log_var, min=-1e+6)
        z = z.reshape(-1, 1, 28, 28)
        return z, mu, log_var
    
    def generate(self, mu, log_var) :
        z = self.reparameterization_trick(mu, log_var)
        z = self.decoder(z)
        z = z.reshape(-1, 1, 28, 28)
        return z
    
    def crossentropy(self, output, target) : #尤度
        # print(output)
        # print(torch.isnan(torch.sum((target + 1) * torch.log(output + 1 + 1e-8) + (509 - target) * torch.log(509 - output + 1e-8))).any())
        return torch.sum(target * torch.log(output + 1e-8) + (1 - target) * torch.log(1 - output + 1e-8))
        # return (target + 1) * torch.log(output + 1 + 1e-8) + (509 - target) * torch.log(509 - output + 1e-8)
    
    def loss_reconstruction(self, output, target) :
        reconst = self.crossentropy(output, target)
        # reconst.clamp_(min=1e-6, max=1e6)
        return -torch.sum(reconst)
    
    def loss_kl_divergence(self, mu, log_var) : #非負

        kl = -0.5 * torch.sum(1 + log_var - mu**2 - torch.exp(log_var))
        return kl

    def loss(self, output, target, mu, log_var) :
        return self.loss_reconstruction(output, target) + self.loss_kl_divergence(mu, log_var)
    