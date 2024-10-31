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
    """
    Builds a vocabulary from a list of SMILES strings.

    Args:
        smiles_list (list of str): List of SMILES strings.

    Returns:
        tuple: A tuple containing:
            - vocab (list of str): List of characters in the vocabulary including padding, start, and end tokens.
            - char_to_idx (dict): Dictionary mapping characters to their corresponding indices in the vocabulary.
            - idx_to_char (dict): Dictionary mapping indices to their corresponding characters in the vocabulary.
            - start_token (str): The start token character.
            - end_token (str): The end token character.
            - pad_token (str): The padding token character.
    """
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
    """
    Encodes a SMILES string into a list of indices based on a character-to-index mapping.

    Parameters:
    smiles (str): The SMILES string to encode.
    char_to_idx (dict): A dictionary mapping characters to their corresponding indices.
    max_length (int): The maximum length of the encoded SMILES string. If the encoded string is shorter, it will be padded.
    start_token (str): The token to prepend to the SMILES string.
    end_token (str): The token to append to the SMILES string.
    pad_token (str): The token used for padding the encoded SMILES string to the maximum length.

    Returns:
    list: A list of indices representing the encoded SMILES string, padded to the specified maximum length.
    """
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
    """
    Encoder module for a Variational Autoencoder (VAE) designed for drug molecule generation.

    Args:
        vocab_size (int): Size of the vocabulary.
        embed_size (int): Size of the embedding vectors.
        latent_dim (int): Dimensionality of the latent space.

    Attributes:
        embedding (nn.Embedding): Embedding layer that converts input tokens to dense vectors.
        conv1 (nn.Conv1d): First 1D convolutional layer.
        conv2 (nn.Conv1d): Second 1D convolutional layer.
        conv3 (nn.Conv1d): Third 1D convolutional layer.
        relu (nn.ReLU): ReLU activation function.
        pool (nn.AdaptiveMaxPool1d): Adaptive max pooling layer.
        fc_mu (nn.Linear): Fully connected layer to compute the mean of the latent space.
        fc_logvar (nn.Linear): Fully connected layer to compute the log variance of the latent space.

    Methods:
        forward(x):
            Forward pass through the encoder network.

            Args:
                x (torch.Tensor): Input tensor of shape [batch_size, seq_len].

            Returns:
                mu (torch.Tensor): Mean of the latent space of shape [batch_size, latent_dim].
                logvar (torch.Tensor): Log variance of the latent space of shape [batch_size, latent_dim].
    """

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
    """
    A GRU-based decoder for generating sequences from latent vectors.

    Args:
        vocab_size (int): Size of the vocabulary.
        embed_size (int): Dimensionality of the embeddings.
        latent_dim (int): Dimensionality of the latent space.
        max_length (int): Maximum length of the generated sequences.

    Attributes:
        latent_dim (int): Dimensionality of the latent space.
        max_length (int): Maximum length of the generated sequences.
        embedding (nn.Embedding): Embedding layer for input sequences.
        gru (nn.GRU): GRU layer for sequence generation.
        fc_out (nn.Linear): Fully connected layer to map GRU outputs to vocabulary size.

    Methods:
        forward(z, target_seq):
            Generates sequences from the latent vector and target sequence.

            Args:
                z (torch.Tensor): Latent vector.
                target_seq (torch.Tensor): Target sequence for teacher forcing.

            Returns:
                torch.Tensor: Output logits for each token in the sequence.
    """

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
    """
    A neural network module for predicting properties from latent vectors.

    Args:
        latent_dim (int): The dimensionality of the latent space.

    Methods:
        forward(z):
            Forward pass through the network.
            Args:
                z (torch.Tensor): A tensor containing the latent vectors.
            Returns:
                torch.Tensor: A tensor containing the predicted properties.
    """

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
    """
    Variational Autoencoder (VAE) class for drug molecule generation.
    Attributes:
        encoder (Encoder): The encoder network that maps input sequences to latent space.
        decoder (Decoder): The decoder network that reconstructs sequences from latent space.
        property_predictor (PropertyPredictor): The network that predicts properties from latent space.
    Methods:
        __init__(vocab_size, embed_size, latent_dim, max_length):
            Initializes the VAE with the given parameters.
        reparameterize(mu, logvar):
            Applies the reparameterization trick to sample from the latent space.
        forward(x):
            Performs a forward pass through the VAE, encoding the input, sampling from the latent space,
            decoding the latent representation, and predicting properties.
    """

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
    """
    Computes the loss for the VAE model, which includes reconstruction loss, KL divergence loss, and property prediction loss.
    Args:
        recon_x (torch.Tensor): The reconstructed output from the decoder.
        x (torch.Tensor): The original input sequence.
        mu (torch.Tensor): The mean of the latent variable distribution.
        logvar (torch.Tensor): The log variance of the latent variable distribution.
        property_pred (torch.Tensor): The predicted property values from the model.
        property_true (torch.Tensor): The true property values.
    Returns:
        torch.Tensor: The computed loss value.
    """

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
    """
    Optimizes a latent vector to maximize or match a target property using gradient descent.
    Args:
        model (torch.nn.Module): The model containing the property predictor.
        initial_z (torch.Tensor): The initial latent vector to be optimized.
        num_steps (int, optional): The number of optimization steps. Default is 100.
        lr (float, optional): The learning rate for the optimizer. Default is 1e-2.
        target_property (torch.Tensor, optional): The target property value to match. If None, the function will maximize the property.
    Returns:
        torch.Tensor: The optimized latent vector.
    """

    z = initial_z.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([z], lr=lr)
    for step in range(num_steps):  # noqa: B007
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
    """
    Decodes a latent vector sample into a SMILES string using the provided VAE model.
    Args:
        model (torch.nn.Module): The VAE model used for decoding.
        z (torch.Tensor): The latent vector sample to decode.
    Returns:
        str: The decoded SMILES string.
    """

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
