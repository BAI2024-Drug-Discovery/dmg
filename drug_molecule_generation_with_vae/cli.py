import click
from rdkit import RDLogger

from drug_molecule_generation_with_vae.analyze import analyze_generated_molecules
from drug_molecule_generation_with_vae.generate import generate as generate_molecules
from drug_molecule_generation_with_vae.train import train as train_model

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
@click.option('--output_path', required=True, help='Path to save the generated molecules CSV file')
def generate(num_molecules, model_dir, output_path):
    """Generate molecules using the trained model."""
    generate_molecules(num_molecules, model_dir, output_path)


@cli.command()
@click.option(
    '--generated_data_path', required=True, help='Path to the generated molecules CSV file'
)
@click.option('--training_data_path', required=True, help='Path to the training data CSV file')
def analyze(generated_data_path, training_data_path):
    """Analyze generated molecules."""
    analyze_generated_molecules(generated_data_path, training_data_path)


if __name__ == '__main__':
    cli()
