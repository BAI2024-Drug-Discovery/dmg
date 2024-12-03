import os

import torch
from rdkit import Chem
from rdkit.Chem import Descriptors

from .models.vae import VAE
from .utils.compute_properties import compute_property
from .utils.config import Config
from .utils.optimization import decode_latent_vector_sample, optimize_latent_vector


def generate(num_molecules, model_dir) -> dict:
    config = Config()
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
    model.load_state_dict(torch.load(os.path.join(model_dir, 'vae_model.pth'), weights_only=True))
    initial_z = torch.randn(1, config.latent_dim).to(config.device)
    optimized_z = optimize_latent_vector(model, initial_z, num_steps=100, lr=1e-2)
    data = {'Generated_SMILES': [], 'Validity': [], 'QED': []}
    for _ in range(num_molecules):
        smiles = decode_latent_vector_sample(
            model, optimized_z, char_to_idx, idx_to_char, start_token, end_token, config.device
        )
        mol = Chem.MolFromSmiles(smiles)
        valid = 'Valid' if mol else 'Invalid'
        properties = compute_property(smiles)
        data['Generated_SMILES'].append(smiles)
        data['Validity'].append(valid)
        data['QED'].append(properties)

    return data
