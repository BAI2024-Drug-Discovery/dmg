import torch


def build_vocab(smiles_list):
    charset = set()
    for smiles in smiles_list:
        for char in smiles:
            charset.add(char)
    start_token = '^'
    end_token = '$'
    pad_token = ' '
    charset = sorted(list(charset))
    vocab = [pad_token, start_token, end_token] + charset
    char_to_idx = {char: idx for idx, char in enumerate(vocab)}
    idx_to_char = {idx: char for idx, char in enumerate(vocab)}
    return vocab, char_to_idx, idx_to_char, start_token, end_token, pad_token


def encode_smiles(smiles, char_to_idx, max_length, start_token, end_token, pad_token):
    smiles = start_token + smiles + end_token
    smiles_idx = [char_to_idx[char] for char in smiles]
    padding = [char_to_idx[pad_token]] * (max_length - len(smiles_idx))
    smiles_idx += padding
    return smiles_idx
