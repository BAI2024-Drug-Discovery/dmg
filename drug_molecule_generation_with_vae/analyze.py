import os

import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors


def calculate_qed(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Descriptors.qed(mol)


def calculate_logp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Descriptors.MolLogP(mol)


def analyze_generated_molecules(generated_data_path, training_data_path):
    # Load generated molecules
    df_generated = pd.read_csv(generated_data_path)
    df_generated['QED'] = df_generated['Generated_SMILES'].apply(calculate_qed)
    df_generated['LogP'] = df_generated['Generated_SMILES'].apply(calculate_logp)

    # Load training data
    df_training = pd.read_csv(training_data_path)
    df_training['QED'] = df_training['SMILES'].apply(calculate_qed)
    df_training['LogP'] = df_training['SMILES'].apply(calculate_logp)

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
    output += f'\nQED KS test statistic: {ks_stat_qed:.4f}, p-value: {p_value_qed:.4e}'
    output += f'\nLogP KS test statistic: {ks_stat_logp:.4f}, p-value: {p_value_logp:.4e}'

    print(output)

    # Write output to a text file
    output_dir = os.path.dirname(generated_data_path)
    with open(os.path.join(output_dir, 'analysis_results.txt'), 'w') as f:
        f.write(output)

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
