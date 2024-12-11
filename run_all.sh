#!/bin/bash


DATA_PATH="training_data/250k_rndm_zinc_drugs_clean_3.smi"
MODEL_DIR="output/vae_model"
GENERATED_DATA_PATH="output/generated_molecules.csv"

# Step 1: Train the model
dmg smilesvae train --data_path $DATA_PATH --output_dir $MODEL_DIR

# Step 2: Generate molecules
dmg smilesvae generate --num_molecules 500 --model_dir $MODEL_DIR --output_path $GENERATED_DATA_PATH

# Step 3: Analyze generated molecules
dmg smilesvae analyze --generated_data_path $GENERATED_DATA_PATH --training_data_path $DATA_PATH