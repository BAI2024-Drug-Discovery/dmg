#!/bin/bash


DATA_PATH="test_files/250k_rndm_zinc_drugs_clean_3.smi"
MODEL_DIR="output/vae_model"
GENERATED_DATA_PATH="output/generated_molecules.csv"

# Step 1: Train the model
drug_molecule_generation_with_vae smiles train --data_path $DATA_PATH --output_dir $MODEL_DIR

# Step 2: Generate molecules
drug_molecule_generation_with_vae smiles generate --num_molecules 500 --model_dir $MODEL_DIR --output_path $GENERATED_DATA_PATH

# Step 3: Analyze generated molecules
drug_molecule_generation_with_vae smiles analyze --generated_data_path $GENERATED_DATA_PATH --training_data_path $DATA_PATH