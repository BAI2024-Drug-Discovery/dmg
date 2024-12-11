from rdkit import Chem
from rdkit.Chem import Descriptors

PROPERTY_FUNCTION = Descriptors.qed


def compute_property(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return PROPERTY_FUNCTION(mol)
