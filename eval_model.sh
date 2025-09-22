
#!/bin/bash

cd /data/chatgpt-training-slc-a100/data/baliao/dynamic_filter/00_start/rlas

# Configuration
K=32
world_size=8

# Model and dataset arrays
models=()
base_model_path="/mnt/nushare2/data/baliao/dynamic_filter/00_start/Reinforceflow/GRPO-Llama-3.2-3B-Instruct-n4"
data_path="/mnt/nushare2/data/baliao/dynamic_filter/data"

chown -R 110541254:110541254 ${base_model_path}

# Generate model paths for global_step_20 to global_step_220 (increment by 20)
for step in $(seq 50 50 500); do
    models+=("$base_model_path/global_step_$step/merged")
done

datasets=("math500" "minerva_math" "olympiadbench" "aime_hmmt_brumo_cmimc_amc23")

# Loop through models and datasets
for model_name in "${models[@]}"; do
    echo "Testing model: $model_name"
    
    for dataset in "${datasets[@]}"; do
        echo "Testing dataset: $dataset"
        
        # Create model/dataset specific output directory
        # Extract global_step_X/merged from the full path
        output_dir=${model_name}/../${dataset}
        mkdir -p ${output_dir}
        
        echo "Output directory: $output_dir"
        
        # Generate data in parallel
        echo "Starting parallel data generation..."
        # we use gpu 4,5,6,7
        for i in 0 1 2 3 4 5 6 7; do
            CUDA_VISIBLE_DEVICES=$i python3 eval_benchmark/gen_data.py \
                --local_index $i \
                --my_world_size $world_size \
                --model_name_or_path "$model_name" \
                --output_dir "$output_dir/" \
                --K $K \
                --use_local_cached_data True \
                --dataset_name_or_path ${data_path}/${dataset} &
        done
        
        # Wait for all parallel processes to complete
        wait
        echo "Data generation completed."
        
        # Merge the generated data
        echo "Merging data..."
        python3 eval_benchmark/merge_data.py \
            --base_path "$output_dir/" \
            --output_dir "$output_dir/merged_data.jsonl" \
            --num_datasets $world_size
        
        if [ $? -ne 0 ]; then
            echo "Error: Failed to merge data for $model_name on $dataset"
            continue
        fi
        
        # Compute scores
        echo "Computing scores..."
        python3 eval_benchmark/compute_score.py \
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

chown -R 110541254:110541254 ${base_model_path}
