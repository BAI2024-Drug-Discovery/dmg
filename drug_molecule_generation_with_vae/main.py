import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from torch.utils.data import DataLoader, TensorDataset

# Disable RDKit warnings
RDLogger.DisableLog('rdApp.*')

# 1. Data Loading and SMILES Processing
smiles_list = pd.read_csv(
    'drug_molecule_generation_with_vae/data/250k_rndm_zinc_drugs_clean_3.csv'
)['SMILES'].tolist()

# Filter out invalid SMILES
smiles_list = [smiles for smiles in smiles_list if Chem.MolFromSmiles(smiles) is not None]


# Build vocabulary
def build_vocab(smiles_list):
    charset = set()
    for smiles in smiles_list:
        for char in smiles:
            charset.add(char)
    # Define start, end, and pad tokens as single characters
    start_token = '^'
    end_token = '$'
    pad_token = ' '  # Using space character as padding
    charset = sorted(list(charset))
    vocab = [pad_token, start_token, end_token] + charset
    char_to_idx = {char: idx for idx, char in enumerate(vocab)}
    idx_to_char = {idx: char for idx, char in enumerate(vocab)}
    return vocab, char_to_idx, idx_to_char, start_token, end_token, pad_token


vocab, char_to_idx, idx_to_char, start_token, end_token, pad_token = build_vocab(smiles_list)
vocab_size = len(vocab)

# Encode SMILES
max_length = max([len(smiles) for smiles in smiles_list]) + 2  # +2 for start and end tokens


def encode_smiles(smiles, char_to_idx, max_length, start_token, end_token, pad_token):
    smiles = start_token + smiles + end_token
    smiles_idx = [char_to_idx[char] for char in smiles]
    padding = [char_to_idx[pad_token]] * (max_length - len(smiles_idx))
    smiles_idx += padding
    return smiles_idx


encoded_smiles = [
    encode_smiles(smiles, char_to_idx, max_length, start_token, end_token, pad_token)
    for smiles in smiles_list
]
encoded_smiles = torch.tensor(encoded_smiles, dtype=torch.long)


# Compute molecular weight as property
def compute_property(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        mw = Descriptors.MolWt(mol)
        return mw
    else:
        return 0.0


properties = [compute_property(smiles) for smiles in smiles_list]
properties = np.array(properties)
mean_prop = np.mean(properties)
std_prop = np.std(properties)
properties = (properties - mean_prop) / std_prop
properties = torch.tensor(properties, dtype=torch.float32)

# Create dataset and dataloader
dataset = TensorDataset(encoded_smiles, properties)
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 2. Model Definitions


# Encoder
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, latent_dim):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=char_to_idx[pad_token])
        self.conv1 = nn.Conv1d(embed_size, 9, kernel_size=9, padding=4)
        self.conv2 = nn.Conv1d(9, 9, kernel_size=9, padding=4)
        self.conv3 = nn.Conv1d(9, 10, kernel_size=11, padding=5)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc_mu = nn.Linear(10, latent_dim)
        self.fc_logvar = nn.Linear(10, latent_dim)

    def forward(self, x):
        x = self.embedding(x)  # [batch_size, seq_len, embed_size]
        x = x.transpose(1, 2)  # [batch_size, embed_size, seq_len]
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pool(x).squeeze(2)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


# Decoder
class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, latent_dim, max_length):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.max_length = max_length
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=char_to_idx[pad_token])
        self.gru = nn.GRU(embed_size, latent_dim, batch_first=True)
        self.fc_out = nn.Linear(latent_dim, vocab_size)

    def forward(self, z, target_seq):
        embeddings = self.embedding(target_seq)
        hidden = z.unsqueeze(0)
        output, hidden = self.gru(embeddings, hidden)
        output = self.fc_out(output)
        return output


# Property Predictor
class PropertyPredictor(nn.Module):
    def __init__(self, latent_dim):
        super(PropertyPredictor, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)

    def forward(self, z):
        x = self.relu(self.fc1(z))
        x = self.fc2(x)
        return x.squeeze(1)


# VAE Model
class VAE(nn.Module):
    def __init__(self, vocab_size, embed_size, latent_dim, max_length):
        super(VAE, self).__init__()
        self.encoder = Encoder(vocab_size, embed_size, latent_dim)
        self.decoder = Decoder(vocab_size, embed_size, latent_dim, max_length)
        self.property_predictor = PropertyPredictor(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        target_seq = x[:, :-1]
        output = self.decoder(z, target_seq)
        property_pred = self.property_predictor(z)
        return output, mu, logvar, property_pred


# Loss Function
def loss_function(recon_x, x, mu, logvar, property_pred, property_true):
    batch_size = x.size(0)
    recon_x = recon_x.view(-1, vocab_size)
    x = x[:, 1:].contiguous().view(-1)
    recon_loss = nn.functional.cross_entropy(
        recon_x, x, ignore_index=char_to_idx[pad_token], reduction='sum'
    )
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    property_loss = nn.functional.mse_loss(property_pred, property_true, reduction='sum')
    return (recon_loss + kl_loss + property_loss) / batch_size


# 3. Training Loop
embed_size = 128
latent_dim = 56
max_length = max_length - 1  # Adjusted for decoder
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = VAE(vocab_size, embed_size, latent_dim, max_length).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in dataloader:
        x_batch, property_batch = batch
        x_batch = x_batch.to(device)
        property_batch = property_batch.to(device)
        optimizer.zero_grad()
        output, mu, logvar, property_pred = model(x_batch)
        loss = loss_function(output, x_batch, mu, logvar, property_pred, property_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}')


# 4. Latent Space Optimization
def optimize_latent_vector(model, initial_z, num_steps=100, lr=1e-2, target_property=None):
    z = initial_z.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([z], lr=lr)
    for step in range(num_steps):
        optimizer.zero_grad()
        property_pred = model.property_predictor(z)
        if target_property is not None:
            loss = (property_pred - target_property).pow(2)
        else:
            loss = -property_pred
        loss.backward()
        optimizer.step()
    return z.detach()


def decode_latent_vector_sample(model, z):
    model.eval()
    with torch.no_grad():
        batch_size = z.size(0)
        hidden = z.unsqueeze(0)
        input_token = torch.full((batch_size, 1), char_to_idx[start_token], dtype=torch.long).to(
            device
        )
        embeddings = model.decoder.embedding(input_token)
        outputs = []
        for _ in range(model.decoder.max_length):
            output, hidden = model.decoder.gru(embeddings, hidden)
            output = model.decoder.fc_out(output.squeeze(1))
            probs = nn.functional.softmax(output, dim=1)
            m = torch.distributions.Categorical(probs)
            topi = m.sample()
            token = topi.item()
            if token == char_to_idx[end_token]:
                break
            outputs.append(token)
            embeddings = model.decoder.embedding(topi.unsqueeze(0))
        decoded_smiles = [idx_to_char[idx] for idx in outputs]
        smiles = ''.join(decoded_smiles)
    return smiles


# Generate Molecules
initial_z = torch.randn(1, latent_dim).to(device)
optimized_z = optimize_latent_vector(model, initial_z, num_steps=100, lr=1e-2)

print('Generated_SMILES,Validity,Molecular_Weight')
for _ in range(10):
    smiles = decode_latent_vector_sample(model, optimized_z)
    mol = Chem.MolFromSmiles(smiles)
    valid = 'Valid' if mol else 'Invalid'
    if mol:
        mw = Descriptors.MolWt(mol)
    else:
        mw = 0.0
    print(f'{smiles},{valid},{mw}')
