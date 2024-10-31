# Import necessary libraries
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from rdkit import Chem
from rdkit.Chem import Descriptors
from torch.utils.data import DataLoader, Dataset

# Set random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 1. Data Preparation

# Load the dataset
data = pd.read_csv(
    'drug_molecule_generation_with_vae/data/250k_rndm_zinc_drugs_clean_3.csv'
)  # Replace with your CSV file path


# Filter out any invalid SMILES
def is_valid_smiles(s):
    return Chem.MolFromSmiles(s) is not None


data = data[data['SMILES'].apply(is_valid_smiles)].reset_index(drop=True)

# Extract SMILES and properties
smiles_list = data['SMILES'].values
logp_list = data['logP'].values
qed_list = data['QED'].values
sas_list = data['SAS'].values

# Build a character set for SMILES tokens
from collections import OrderedDict


def tokenize_smiles(s):
    tokens = []
    i = 0
    while i < len(s):
        if s[i : i + 2] in ['Br', 'Cl']:
            tokens.append(s[i : i + 2])
            i += 2
        else:
            tokens.append(s[i])
            i += 1
    return tokens


# Create vocabulary
all_chars = set()
for s in smiles_list:
    tokens = tokenize_smiles(s)
    all_chars.update(tokens)

# Create mappings
char2idx = {c: i + 1 for i, c in enumerate(sorted(all_chars))}
char2idx['<pad>'] = 0  # Padding token
char2idx['<start>'] = len(char2idx)
char2idx['<end>'] = len(char2idx)
idx2char = {i: c for c, i in char2idx.items()}
vocab_size = len(char2idx)

# Max sequence length
max_len = max([len(tokenize_smiles(s)) for s in smiles_list]) + 2  # for <start> and <end> tokens


def smiles_to_seq(s):
    tokens = tokenize_smiles(s)
    seq = [char2idx['<start>']] + [char2idx[c] for c in tokens] + [char2idx['<end>']]
    seq += [char2idx['<pad>']] * (max_len - len(seq))
    return seq


# Create Dataset and DataLoader
class SmilesDataset(Dataset):
    def __init__(self, smiles_list, logp_list, qed_list, sas_list):
        self.smiles_list = smiles_list
        self.logp_list = logp_list
        self.qed_list = qed_list
        self.sas_list = sas_list

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        seq = smiles_to_seq(smiles)
        properties = np.array(
            [self.logp_list[idx], self.qed_list[idx], self.sas_list[idx]], dtype=np.float32
        )
        return torch.tensor(seq, dtype=torch.long), torch.tensor(properties, dtype=torch.float32)


# Split data into training and validation sets
from sklearn.model_selection import train_test_split

train_smiles, val_smiles, train_logp, val_logp, train_qed, val_qed, train_sas, val_sas = (
    train_test_split(smiles_list, logp_list, qed_list, sas_list, test_size=0.1, random_state=seed)
)

train_dataset = SmilesDataset(train_smiles, train_logp, train_qed, train_sas)
val_dataset = SmilesDataset(val_smiles, val_logp, val_qed, val_sas)

batch_size = 64

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

# 2. Model Architecture - MolVAE


# Encoder
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, latent_dim, property_dim):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=char2idx['<pad>'])
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc_mu = nn.Linear(hidden_dim * 2 + property_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2 + property_dim, latent_dim)

    def forward(self, x, properties):
        embedded = self.embedding(x)
        _, h = self.gru(embedded)
        h = torch.cat((h[0], h[1]), dim=1)  # Concatenate forward and backward hidden states
        h = torch.cat((h, properties), dim=1)  # Concatenate properties
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


# Decoder
class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, latent_dim, property_dim):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=char2idx['<pad>'])
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.latent_to_hidden = nn.Linear(latent_dim + property_dim, hidden_dim)

    def forward(self, x, z):
        embedded = self.embedding(x)
        h0 = torch.tanh(self.latent_to_hidden(z)).unsqueeze(0)  # Initial hidden state
        output, _ = self.gru(embedded, h0)
        logits = self.fc_out(output)
        return logits


# Reparameterization Trick
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


# 3. Training Process

# Loss Function
criterion = nn.CrossEntropyLoss(ignore_index=char2idx['<pad>'])

# Model Initialization
embedding_dim = 128
hidden_dim = 256
latent_dim = 128
property_dim = 3  # logP, QED, SAS

encoder = Encoder(vocab_size, embedding_dim, hidden_dim, latent_dim, property_dim).to(device)
decoder = Decoder(vocab_size, embedding_dim, hidden_dim, latent_dim, property_dim).to(device)

# Optimizer
params = list(encoder.parameters()) + list(decoder.parameters())
optimizer = optim.Adam(params, lr=0.001)

# Training Loop
num_epochs = 20
beta = 1.0  # Weight for KL divergence

for epoch in range(num_epochs):
    encoder.train()
    decoder.train()
    total_loss = 0
    for seqs, properties in train_loader:
        seqs = seqs.to(device)  # Move to device
        properties = properties.to(device)  # Move to device

        # Prepare inputs and targets
        input_seq = seqs[:, :-1]
        target_seq = seqs[:, 1:]

        # Encode
        mu, logvar = encoder(input_seq, properties)
        z = reparameterize(mu, logvar)
        z = torch.cat((z, properties), dim=1)  # Concatenate properties to latent vector

        # Decode
        logits = decoder(input_seq, z)

        # Compute reconstruction loss
        logits = logits.view(-1, vocab_size)
        target_seq = target_seq.contiguous().view(-1)
        recon_loss = criterion(logits, target_seq)

        # Compute KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / seqs.size(0)

        loss = recon_loss + beta * kl_loss

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

# 4. Sampling New Molecules

encoder.eval()
decoder.eval()


def sample_molecules(num_samples, properties):
    generated_smiles = []
    with torch.no_grad():
        for _ in range(num_samples):
            # Sample latent vector from standard normal distribution
            z = torch.randn(1, latent_dim).to(device)
            prop_tensor = torch.tensor(properties, dtype=torch.float32).unsqueeze(0).to(device)
            z = torch.cat((z, prop_tensor), dim=1)

            # Generate sequences
            generated_seq = [char2idx['<start>']]
            input_seq = torch.tensor([[char2idx['<start>']]], dtype=torch.long).to(device)

            hidden = torch.tanh(decoder.latent_to_hidden(z)).unsqueeze(0)

            for _ in range(max_len):
                embedded = decoder.embedding(input_seq)
                output, hidden = decoder.gru(embedded, hidden)
                logits = decoder.fc_out(output.squeeze(1))
                prob = torch.softmax(logits, dim=1)
                next_token = torch.multinomial(prob, num_samples=1).item()
                if next_token == char2idx['<end>'] or next_token == char2idx['<pad>']:
                    break
                generated_seq.append(next_token)
                input_seq = torch.tensor([[next_token]], dtype=torch.long).to(device)

            # Convert token indices back to SMILES
            tokens = [idx2char[idx] for idx in generated_seq[1:]]  # Exclude <start> token
            smiles = ''.join(tokens)
            generated_smiles.append(smiles)
    return generated_smiles


# Example of generating molecules with desired properties
desired_properties = [2.0, 0.8, 3.0]  # Example values for logP, QED, SAS
generated_smiles = sample_molecules(10, desired_properties)

# 5. Evaluation Metrics

# Validity
valid_smiles = []
for s in generated_smiles:
    mol = Chem.MolFromSmiles(s)
    if mol:
        valid_smiles.append(s)

validity = len(valid_smiles) / len(generated_smiles)
print(f'Validity: {validity * 100:.2f}%')

# Uniqueness
unique_smiles = set(valid_smiles)
uniqueness = len(unique_smiles) / len(valid_smiles) if valid_smiles else 0
print(f'Uniqueness: {uniqueness * 100:.2f}%')

# Novelty
training_set = set(smiles_list)
novel_smiles = unique_smiles - training_set
novelty = len(novel_smiles) / len(unique_smiles) if unique_smiles else 0
print(f'Novelty: {novelty * 100:.2f}%')


# Property Evaluation
def calculate_sas(mol):
    # Placeholder for SAS calculation
    # Implement SAS calculation or use an external library
    return 3.0  # Example value


for s in valid_smiles:
    mol = Chem.MolFromSmiles(s)
    if mol:
        logp = Descriptors.MolLogP(mol)
        qed = Descriptors.qed(mol)
        sas = calculate_sas(mol)
        print(f'SMILES: {s}, logP: {logp:.2f}, QED: {qed:.2f}, SAS: {sas:.2f}')
