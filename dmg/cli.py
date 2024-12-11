import click
from rdkit import RDLogger

from dmg.smiles_vae.analyze import (
    analyze as smiles_analyze,
)
from dmg.smiles_vae.generate import generate as smiles_generate
from dmg.smiles_vae.train import train as smiles_train

RDLogger.DisableLog('rdApp.*')


@click.group()
def cli():
    pass


@click.group()
def smilesvae():
    """Commands related to SMILES representation of molecules."""
    pass


@smilesvae.command()
@click.option('--data_path', required=True, help='Path to the molecules CSV file')
@click.option(
    '--output_dir', required=True, help='Directory to save the trained model and information'
)
def train(data_path, output_dir):
    """Train a VAE model."""
    smiles_train(data_path, output_dir)


@smilesvae.command()
@click.option('--num_molecules', default=100, help='Number of molecules to generate')
@click.option('--model_dir', required=True, help='Directory containing the model and information')
@click.option('--output_path', required=True, help='Path to save the generated molecules CSV file')
def generate(num_molecules, model_dir, output_path):
    """Generate molecules using the trained model."""
    smiles_generate(num_molecules, model_dir, output_path)


@smilesvae.command()
@click.option(
    '--generated_data_path', required=True, help='Path to the generated molecules CSV file'
)
@click.option('--training_data_path', required=True, help='Path to the training data CSV file')
def analyze(generated_data_path, training_data_path):
    """Analyze generated molecules."""
    smiles_analyze(generated_data_path, training_data_path)


cli.add_command(smilesvae)

if __name__ == '__main__':
    cli()
