import pandas as pd
from rdkit import Chem


def load_smiles(file_path):
    smiles_list = pd.read_csv(file_path)['SMILES'].tolist()
    smiles_list = [smiles for smiles in smiles_list if Chem.MolFromSmiles(smiles) is not None]
    return smiles_list
