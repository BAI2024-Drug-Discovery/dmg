import torch.nn as nn


class PropertyPredictor(nn.Module):
    def __init__(self, latent_dim):
        super(PropertyPredictor, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)

    def forward(self, z):
        x = self.relu(self.fc1(z))
        x = self.fc2(x)
        return x.squeeze(1)
