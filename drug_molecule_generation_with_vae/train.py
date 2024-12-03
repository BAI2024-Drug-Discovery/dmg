import os

import torch
from torch.utils.data import DataLoader, TensorDataset

from .data.data_loader import load_smiles
from .models.vae import VAE
from .utils.compute_properties import compute_property
from .utils.config import Config
from .utils.loss_function import loss_function
from .utils.smiles_processing import build_vocab, encode_smiles


def train(data_path, output_dir):
    config = Config()
    smiles_list = load_smiles(data_path)
    property_list = [compute_property(smiles) for smiles in smiles_list]
    vocab, char_to_idx, idx_to_char, start_token, end_token, pad_token = build_vocab(smiles_list)
    vocab_size = len(vocab)
    max_length = max([len(smiles) for smiles in smiles_list]) + 2
    encoded_smiles = [
        encode_smiles(smiles, char_to_idx, max_length, start_token, end_token, pad_token)
        for smiles in smiles_list
    ]
    encoded_smiles = torch.tensor(encoded_smiles, dtype=torch.long)
    properties = torch.tensor(property_list, dtype=torch.float32)
    dataset = TensorDataset(encoded_smiles, properties)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    model = VAE(
        vocab_size, config.embed_size, config.latent_dim, max_length - 1, char_to_idx[pad_token]
    ).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            x_batch, property_batch = batch
            x_batch = x_batch.to(config.device)
            property_batch = property_batch.to(config.device)
            optimizer.zero_grad()
            output, mu, logvar, property_pred = model(x_batch)
            loss = loss_function(
                output,
                x_batch,
                mu,
                logvar,
                property_pred,
                property_batch,
                vocab_size,
                char_to_idx[pad_token],
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}')
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, 'vae_model.pth'))
    torch.save(
        {
            'vocab': vocab,
            'char_to_idx': char_to_idx,
            'idx_to_char': idx_to_char,
            'start_token': start_token,
            'end_token': end_token,
            'pad_token': pad_token,
            'max_length': max_length,
        },
        os.path.join(output_dir, 'model_info.pth'),
    )
    print(f'Model and information saved in {output_dir}')
