import pandas as pd
from rdkit import Chem

from ..utils.compute_properties import compute_property


def load_smiles(file_path):
    """Load SMILES from a SMI file, where each line is a SMILES string.
    Args:
        file_path (str): Path to the SMI file.
    Returns:
        list: List of SMILES strings.

    """

    with open(file_path, 'r') as f:
        smiles_list = f.read().splitlines()

    smiles_list = [smiles for smiles in smiles_list if Chem.MolFromSmiles(smiles) is not None]
    smiles_list = [smiles for smiles in smiles_list if compute_property(smiles) is not None]
    return smiles_list
