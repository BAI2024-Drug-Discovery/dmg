#!/bin/bash

. /home/marius/uni/drug-molecule-generation-with-VAE/.venv/bin/activate 

DATA_PATH="notebooks/250k_rndm_zinc_drugs_clean_3.csv"
MODEL_DIR="output/vae_model"
GENERATED_DATA_PATH="output/generated_molecules.csv"

# Step 1: Train the model
drug_molecule_generation_with_vae train --data_path $DATA_PATH --output_dir $MODEL_DIR

# Step 2: Generate molecules
drug_molecule_generation_with_vae generate --num_molecules 500 --model_dir $MODEL_DIR --output_path $GENERATED_DATA_PATH

# Step 3: Analyze generated molecules
drug_molecule_generation_with_vae analyze --generated_data_path $GENERATED_DATA_PATH --training_data_path $DATA_PATH