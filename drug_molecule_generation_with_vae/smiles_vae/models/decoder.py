import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, latent_dim, max_length, pad_token_idx):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.max_length = max_length
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_token_idx)
        self.gru = nn.GRU(embed_size, latent_dim, batch_first=True)
        self.fc_out = nn.Linear(latent_dim, vocab_size)

    def forward(self, z, target_seq):
        embeddings = self.embedding(target_seq)
        hidden = z.unsqueeze(0)
        output, hidden = self.gru(embeddings, hidden)
        output = self.fc_out(output)
        return output
