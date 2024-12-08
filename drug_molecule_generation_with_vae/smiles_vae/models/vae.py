import torch
import torch.nn as nn

from .decoder import Decoder
from .encoder import Encoder
from .property_predictor import PropertyPredictor


class VAE(nn.Module):
    def __init__(self, vocab_size, embed_size, latent_dim, max_length, pad_token_idx):
        super(VAE, self).__init__()
        self.encoder = Encoder(vocab_size, embed_size, latent_dim, pad_token_idx)
        self.decoder = Decoder(vocab_size, embed_size, latent_dim, max_length, pad_token_idx)
        self.property_predictor = PropertyPredictor(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        target_seq = x[:, :-1]
        output = self.decoder(z, target_seq)
        property_pred = self.property_predictor(z)
        return output, mu, logvar, property_pred
