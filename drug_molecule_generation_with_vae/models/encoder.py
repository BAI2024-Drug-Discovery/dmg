import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, latent_dim, pad_token_idx):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_token_idx)
        self.conv1 = nn.Conv1d(embed_size, 9, kernel_size=9, padding=4)
        self.conv2 = nn.Conv1d(9, 9, kernel_size=9, padding=4)
        self.conv3 = nn.Conv1d(9, 10, kernel_size=11, padding=5)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc_mu = nn.Linear(10, latent_dim)
        self.fc_logvar = nn.Linear(10, latent_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pool(x).squeeze(2)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
