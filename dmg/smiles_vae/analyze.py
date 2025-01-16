import os

import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors

from dmg.smiles_vae.data.data_loader import load_smiles


def calculate_qed(smiles):
    """
    Calculate the Quantitative Estimate of Drug-likeness (QED) for a given SMILES string.

    Parameters:
    smiles (str): A string representing the SMILES notation of the molecule.

    Returns:
    float or None: The QED value of the molecule if the SMILES string is valid, otherwise None.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Descriptors.qed(mol)


def calculate_logp(smiles):
    """
    Calculate the logarithm of the partition coefficient (logP) for a given SMILES string.

    Parameters:
    smiles (str): A string representing the SMILES notation of the molecule.

    Returns:
    float or None: The logP value of the molecule if the SMILES string is valid, otherwise None.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Descriptors.MolLogP(mol)


def calculate_molecular_weight(smiles):
    """
    Calculate the molecular weight for a given SMILES string.

    Parameters:
    smiles (str): A string representing the SMILES notation of the molecule.

    Returns:
    float or None: The molecular weight of the molecule if the SMILES string is valid, otherwise None.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Descriptors.MolWt(mol)


def is_significant(p_value, alpha=0.05):
    """
    Determine if the p-value is statistically significant.

    Args:
        p_value (float): The p-value from a statistical test.
        alpha (float): The significance level.

    Returns:
        bool: True if the p-value is less than the significance level, False otherwise.
    """
    return p_value < alpha


def analyze(generated_data_path, training_data_path):
    """
    Analyze the generated molecules by comparing them with the training data.

    Args:
        generated_data_path (str): Path to the CSV file containing the generated molecules.
        training_data_path (str): Path to the CSV file containing the training data.

    Returns:
        None
    """
    # Load generated molecules
    df_generated = pd.read_csv(generated_data_path)
    df_generated['QED'] = df_generated['Generated_SMILES'].apply(calculate_qed)
    df_generated['LogP'] = df_generated['Generated_SMILES'].apply(calculate_logp)
    df_generated['MolecularWeight'] = df_generated['Generated_SMILES'].apply(
        calculate_molecular_weight
    )

    # Load training data from the original dataset (smi file)
    training_smiles = load_smiles(training_data_path)
    df_training = pd.DataFrame(training_smiles, columns=['SMILES'])
    df_training['QED'] = df_training['SMILES'].apply(calculate_qed)
    df_training['LogP'] = df_training['SMILES'].apply(calculate_logp)
    df_training['MolecularWeight'] = df_training['SMILES'].apply(calculate_molecular_weight)

    num_generated = len(df_generated)
    output = f'Generated molecules: {num_generated}'

    # Valid generated molecules
    num_valid = len(df_generated[df_generated['Validity'] == 'Valid'])
    valid_percentage = (num_valid / len(df_generated)) * 100
    output += f'\nValid molecules: {num_valid} ({valid_percentage:.2f}%)'

    # Determine novel molecules not in training data
    valid_smiles = df_generated[df_generated['Validity'] == 'Valid']
    training_smiles = set(df_training['SMILES'])
    novel_smiles = valid_smiles[~valid_smiles['Generated_SMILES'].isin(training_smiles)]
    num_novel = len(novel_smiles)
    novel_percentage = (num_novel / num_valid) * 100
    output += f'\nNovel molecules: {num_novel} ({novel_percentage:.2f}%)'

    # Perform statistical tests
    ks_stat_qed, p_value_qed = stats.ks_2samp(
        df_training['QED'].dropna(), df_generated['QED'].dropna()
    )
    ks_stat_logp, p_value_logp = stats.ks_2samp(
        df_training['LogP'].dropna(), df_generated['LogP'].dropna()
    )
    ks_stat_mw, p_value_mw = stats.ks_2samp(
        df_training['MolecularWeight'].dropna(), df_generated['MolecularWeight'].dropna()
    )
    output += f'\nQED KS test statistic: {ks_stat_qed:.4f}, p-value: {p_value_qed:.4e}, significant: {is_significant(p_value_qed)}'
    output += f'\nLogP KS test statistic: {ks_stat_logp:.4f}, p-value: {p_value_logp:.4e}, significant: {is_significant(p_value_logp)}'
    output += f'\nMolecular Weight KS test statistic: {ks_stat_mw:.4f}, p-value: {p_value_mw:.4e}, significant: {is_significant(p_value_mw)}'

    print(output)

    # Write output to a text file
    output_dir = os.path.dirname(generated_data_path)
    with open(os.path.join(output_dir, 'analysis_results.txt'), 'w') as f:
        f.write(output)

    # Plot the statistical test results
    plt.figure(figsize=(10, 6))
    plt.bar(['QED', 'LogP', 'Molecular Weight'], [ks_stat_qed, ks_stat_logp, ks_stat_mw])
    plt.xlabel('Descriptor')
    plt.ylabel('KS Test Statistic')
    plt.title('KS Test Statistics for QED, LogP, and Molecular Weight')
    plt.savefig(os.path.join(output_dir, 'ks_test_statistics.png'))
    plt.close()

    # Plot QED distribution
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df_training['QED'], label='Training Set', fill=True)
    sns.kdeplot(df_generated['QED'].dropna(), label='Generated Molecules', fill=True)
    plt.xlabel('QED')
    plt.ylabel('Density')
    plt.title('QED Distribution')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'qed_distribution.png'))
    plt.close()

    # Plot LogP distribution
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df_training['LogP'], label='Training Set', fill=True)
    sns.kdeplot(df_generated['LogP'].dropna(), label='Generated Molecules', fill=True)
    plt.xlabel('LogP')
    plt.ylabel('Density')
    plt.title('LogP Distribution')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'logp_distribution.png'))
    plt.close()

    # Plot Molecular Weight distribution
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df_training['MolecularWeight'], label='Training Set', fill=True)
    sns.kdeplot(df_generated['MolecularWeight'].dropna(), label='Generated Molecules', fill=True)
    plt.xlabel('Molecular Weight')
    plt.ylabel('Density')
    plt.title('Molecular Weight Distribution')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'molecular_weight_distribution.png'))
    plt.close()
