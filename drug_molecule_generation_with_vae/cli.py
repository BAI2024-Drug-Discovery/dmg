import os

import click
import torch
from data.data_loader import load_smiles
from models.vae import VAE
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from torch.utils.data import DataLoader, TensorDataset
from utils.compute_properties import compute_property
from utils.config import Config
from utils.optimization import decode_latent_vector_sample, optimize_latent_vector
from utils.smiles_processing import build_vocab, encode_smiles

RDLogger.DisableLog('rdApp.*')


def loss_function(recon_x, x, mu, logvar, property_pred, property_true, vocab_size, pad_token_idx):
    batch_size = x.size(0)
    recon_x = recon_x.view(-1, vocab_size)
    x = x[:, 1:].contiguous().view(-1)
    recon_loss = torch.nn.functional.cross_entropy(
        recon_x, x, ignore_index=pad_token_idx, reduction='sum'
    )
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    property_loss = torch.nn.functional.mse_loss(property_pred, property_true, reduction='sum')
    return (recon_loss + kl_loss + property_loss) / batch_size


@click.group()
def cli():
    pass


@cli.command()
@click.option('--data_path', required=True, help='Path to the molecules CSV file')
@click.option(
    '--output_dir', required=True, help='Directory to save the trained model and information'
)
def train(data_path, output_dir):
    config = Config()
    smiles_list = load_smiles(data_path)
    vocab, char_to_idx, idx_to_char, start_token, end_token, pad_token = build_vocab(smiles_list)
    vocab_size = len(vocab)
    max_length = max([len(smiles) for smiles in smiles_list]) + 2

    encoded_smiles = [
        encode_smiles(smiles, char_to_idx, max_length, start_token, end_token, pad_token)
        for smiles in smiles_list
    ]
    encoded_smiles = torch.tensor(encoded_smiles, dtype=torch.long)

    properties = [compute_property(smiles) for smiles in smiles_list]
    properties = torch.tensor(properties, dtype=torch.float32)

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

    # Save the trained model and necessary information
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


@cli.command()
@click.option('--num_molecules', default=100, help='Number of molecules to generate')
@click.option('--model_dir', required=True, help='Directory containing the model and information')
def generate(num_molecules, model_dir):
    config = Config()

    # Load the model and necessary information
    model_info = torch.load(os.path.join(model_dir, 'model_info.pth'), weights_only=True)
    vocab = model_info['vocab']
    char_to_idx = model_info['char_to_idx']
    idx_to_char = model_info['idx_to_char']
    start_token = model_info['start_token']
    end_token = model_info['end_token']
    pad_token = model_info['pad_token']
    max_length = model_info['max_length']
    vocab_size = len(vocab)

    model = VAE(
        vocab_size, config.embed_size, config.latent_dim, max_length - 1, char_to_idx[pad_token]
    ).to(config.device)
    model.load_state_dict(
        torch.load(os.path.join(model_dir, 'vae_model.pth'), weights_only=True)
    )  # Load the trained model

    initial_z = torch.randn(1, config.latent_dim).to(config.device)
    optimized_z = optimize_latent_vector(model, initial_z, num_steps=100, lr=1e-2)

    print('Generated_SMILES,Validity,Molecular_Weight')
    for _ in range(num_molecules):
        smiles = decode_latent_vector_sample(
            model, optimized_z, char_to_idx, idx_to_char, start_token, end_token, config.device
        )
        mol = Chem.MolFromSmiles(smiles)
        valid = 'Valid' if mol else 'Invalid'
        mw = Descriptors.MolWt(mol) if mol else 0.0
        print(f'{smiles},{valid},{mw}')


if __name__ == '__main__':
    cli()
