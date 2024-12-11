# Drug Molecule Generation (DMG)

This project implements a Variational Autoencoder (VAE) for generating drug molecules. The VAE model is designed to encode molecular structures into a latent space and decode them back into molecular structures. Additionally, the model predicts properties of the generated molecules.

## Setup

### Environment

First create new virtual environment.

```bash
python -m venv .venv
sourcen .venv/bin/activate

pip install .
```

## Usage

```bash
dmg --help
```

## Relevant Links

<https://keras.io/examples/generative/molecule_generation/>

<https://huggingface.co/keras-io/drug-molecule-generation-with-VAE>

## Relevant Papers

[Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules](https://pubs.acs.org/doi/10.1021/acscentsci.7b00572)

[MolGAN: An implicit generative model for small molecular graphs](https://arxiv.org/abs/1805.11973)
