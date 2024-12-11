import torch
import torch.optim as optim


def optimize_latent_vector(model, initial_z, num_steps=100, lr=1e-2, target_property=None):
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


def decode_latent_vector_sample(model, z, char_to_idx, idx_to_char, start_token, end_token, device):
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
            probs = torch.nn.functional.softmax(output, dim=1)
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
