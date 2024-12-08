import torch


class Config:
    def __init__(self):
        self.embed_size = 128
        self.latent_dim = 56
        self.max_length = 100  # Adjust as needed
        self.batch_size = 32
        self.num_epochs = 10
        self.learning_rate = 1e-3
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
