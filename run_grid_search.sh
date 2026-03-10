#!/bin/bash

# Hyperparameter Grid Search Script for Qwen-VL Fine-tuning

# Define hyperparameter grid
LEARNING_RATES=(1e-4 5e-5)
LORA_RANKS=(8 16)
LORA_ALPHAS=(8 16) # Must correspond to LORA_RANKS length or use nested loops
EPOCHS=(3 5)

# Fixed parameters
BATCH_SIZE=1
GRAD_ACCUM=8
WARMUP_STEPS=50
LOGGING_STEPS=5
SAVE_STEPS=50
EVAL_STEPS=50

# Ensure logs directory exists
mkdir -p logs
mkdir -p output/grid_search

echo "Starting Hyperparameter Grid Search..."

# Loop through combinations
for LR in "${LEARNING_RATES[@]}"; do
    for i in "${!LORA_RANKS[@]}"; do
        RANK="${LORA_RANKS[$i]}"
        ALPHA="${LORA_ALPHAS[$i]}"
        
        for EP in "${EPOCHS[@]}"; do
            
            # Define output directory for this specific run
            RUN_NAME="lr_${LR}_r_${RANK}_alpha_${ALPHA}_ep_${EP}"
            OUTPUT_DIR="./output/grid_search/${RUN_NAME}"
            LOG_FILE="logs/${RUN_NAME}.log"
            
            echo "============================================================"
            echo "Running experiment: $RUN_NAME"
            echo "Output Dir: $OUTPUT_DIR"
            echo "Log File: $LOG_FILE"
            echo "============================================================"
            
            # Run finetune.py
            python finetune.py \
              --num_train_epochs ${EP} \
              --per_device_train_batch_size ${BATCH_SIZE} \
              --per_device_eval_batch_size ${BATCH_SIZE} \
              --gradient_accumulation_steps ${GRAD_ACCUM} \
              --learning_rate ${LR} \
              --warmup_steps ${WARMUP_STEPS} \
              --logging_steps ${LOGGING_STEPS} \
              --save_steps ${SAVE_STEPS} \
              --eval_steps ${EVAL_STEPS} \
              --output_dir ${OUTPUT_DIR} \
              --use_qlora False \
              --lora_r ${RANK} \
              --lora_alpha ${ALPHA} > "${LOG_FILE}" 2>&1
            
            if [ $? -eq 0 ]; then
                echo "Successfully completed: $RUN_NAME"
            else
                echo "Error occurred in run: $RUN_NAME. Check $LOG_FILE for details."
                # Uncomment the following line to stop the entire grid search on error
                # exit 1 
            fi
            
            echo "------------------------------------------------------------"
            sleep 5 # Small delay between runs
            
        done
    done
done

echo "Grid Search Completed!"
