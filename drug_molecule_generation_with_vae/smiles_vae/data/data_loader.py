import pandas as pd
from rdkit import Chem

from ..utils.compute_properties import compute_property


def load_smiles(file_path):
    smiles_list = pd.read_csv(file_path)['SMILES'].tolist()
    smiles_list = [smiles for smiles in smiles_list if Chem.MolFromSmiles(smiles) is not None]
    smiles_list = [smiles for smiles in smiles_list if compute_property(smiles) is not None]
    return smiles_list


def load_graphs(file_path):
    smiles = load_smiles(file_path)

    graphs = []
    for s in smiles:
        graphs.append(Chem.MolFromSmiles(s))
    return graphs
