# drug-molecule-generation-with-VAE

This project implements a Variational Autoencoder (VAE) for generating drug molecules. The VAE model is designed to encode molecular structures into a latent space and decode them back into molecular structures. Additionally, the model predicts properties of the generated molecules.

## Setup

### Conda Environment

```bash
conda env create -f env.yml
conda activate dmgwvae
```

### Dependencies

- PyTorch
- RDKit
- NumPy
- Pandas

## Usage

```bash
python drug_molecule_generation_with_vae/main.py
```

## Model Architecture

### Encoder

The `Encoder` class encodes input sequences into a latent space.

### Decoder

The `Decoder` class decodes latent vectors back into molecular sequences.

### Property Predictor

The `PropertyPredictor` class predicts properties from latent vectors.

### VAE

The `VAE` class combines the encoder, decoder, and property predictor to form the complete VAE model.

## Relevant Links

https://keras.io/examples/generative/molecule_generation/

https://huggingface.co/keras-io/drug-molecule-generation-with-VAE

## Relevant Papers

[Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules](https://pubs.acs.org/doi/10.1021/acscentsci.7b00572)

[MolGAN: An implicit generative model for small molecular graphs](https://arxiv.org/abs/1805.11973)
