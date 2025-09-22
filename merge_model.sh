#!/bin/bash

models=()
base_model_path="/mnt/nushare2/data/baliao/dynamic_filter/00_start/Reinforceflow/GRPO-Llama-3.2-3B-Instruct-n4"

chown -R 110541254:110541254 ${base_model_path}

for step in $(seq 50 50 500); do
    models+=("$base_model_path/global_step_$step")
done

for model_name in "${models[@]}"; do
    python -m verl.model_merger merge \
        --backend fsdp \
        --local_dir ${model_name}/actor \
        --target_dir ${model_name}/merged
done

chown -R 110541254:110541254 ${base_model_path}