import os

import click
import torch
from data.data_loader import load_smiles
from generate import generate as generate_molecules
from models.vae import VAE
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from torch.utils.data import DataLoader, TensorDataset
from train import train as train_model
from utils.compute_properties import compute_property
from utils.config import Config
from utils.optimization import decode_latent_vector_sample, optimize_latent_vector
from utils.smiles_processing import build_vocab, encode_smiles

RDLogger.DisableLog('rdApp.*')


@click.group()
def cli():
    pass


@cli.command()
@click.option('--data_path', required=True, help='Path to the molecules CSV file')
@click.option(
    '--output_dir', required=True, help='Directory to save the trained model and information'
)
def train(data_path, output_dir):
    train_model(data_path, output_dir)


@cli.command()
@click.option('--num_molecules', default=100, help='Number of molecules to generate')
@click.option('--model_dir', required=True, help='Directory containing the model and information')
def generate(num_molecules, model_dir):
    generate_molecules(num_molecules, model_dir)


if __name__ == '__main__':
    cli()
