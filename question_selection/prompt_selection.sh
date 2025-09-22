#!/bin/bash

# Enter the parent directory
cd "/data/chatgpt/data/baliao/dynamic_filter/00_start/rlas/eval_benchmark"

# Configuration
base_output_dir="/mnt/nushare2/data/baliao/dynamic_filter/00_start/question_selection"
K=16
world_size=24

# Model and dataset arrays
models=("/mnt/nushare2/data/baliao/PLLMs/meta-llama/Llama-3.2-3B-Instruct")
datasets=("/mnt/nushare2/data/baliao/dynamic_filter/data/from_default_filtered_openr1")

# Create base output directory
mkdir -p $base_output_dir

# Loop through models and datasets
for model_name in "${models[@]}"; do
    echo "Testing model: $model_name"
    
    for dataset in "${datasets[@]}"; do
        echo "Testing dataset: $dataset"
        
        # Create model/dataset specific output directory
        output_dir="$base_output_dir"
        mkdir -p "$output_dir"
        
        echo "Output directory: $output_dir"
        
        # Generate data in parallel
        echo "Starting parallel data generation..."
        # we use gpu 4,5,6,7
        for i in 0 1 2 3 4 5 6 7; do
            CUDA_VISIBLE_DEVICES=$i python3 gen_data.py \
                --local_index $i \
                --my_world_size $world_size \
                --model_name_or_path "$model_name" \
                --output_dir "$output_dir/" \
                --K $K \
                --dataset_name_or_path "$dataset" \
                --use_local_cached_data True &
        done
        
        # Wait for all parallel processes to complete
        wait
        echo "Data generation completed."
        
        # Merge the generated data
        echo "Merging data..."
        python3 merge_data.py \
            --base_path "$output_dir/" \
            --output_dir "$output_dir/merged_data.jsonl" \
            --num_datasets $world_size
        
        if [ $? -ne 0 ]; then
            echo "Error: Failed to merge data for $model_name on $dataset"
            continue
        fi
        
        # Compute scores
        echo "Computing scores..."
        python3 compute_score.py \
            --dataset_path "$output_dir/merged_data.jsonl" \
            --record_path "$output_dir/record.txt"
        
        if [ $? -ne 0 ]; then
            echo "Error: Failed to compute scores for $model_name on $dataset"
            continue
        fi
        
        echo "Completed evaluation for $model_name on $dataset"
        echo "Results saved to: $output_dir/record.txt"
        echo "----------------------------------------"
    done
done

echo "All evaluations completed!"