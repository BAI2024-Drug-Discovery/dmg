import torch


def loss_function(recon_x, x, mu, logvar, property_pred, property_true, vocab_size, pad_token_idx):
    batch_size = x.size(0)
    recon_x = recon_x.view(-1, vocab_size)
    x = x[:, 1:].contiguous().view(-1)
    recon_loss = torch.nn.functional.cross_entropy(
        recon_x, x, ignore_index=pad_token_idx, reduction='sum'
    )
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    property_loss = torch.nn.functional.mse_loss(property_pred, property_true, reduction='sum')
    return (recon_loss + kl_loss + property_loss) / batch_size
