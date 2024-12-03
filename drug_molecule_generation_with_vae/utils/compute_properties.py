from rdkit import Chem
from rdkit.Chem import Descriptors


def compute_property(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        mw = Descriptors.MolWt(mol)
        return mw
    else:
        return 0.0
