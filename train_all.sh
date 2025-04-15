#!/bin/bash

# Define possible values for each setting
OS_values=("strang" "lie")
ORDER_values=("ADR" "DRA")
INTEGRATOR_values=("FE" "RK4")
OUTPUT_DIR="/home/ndj376/ADRnet/AdvectionNet/PDEBench_SWE_ADRNet_Pred50/test_results/"
mkdir -p $OUTPUT_DIR

# Loop through all combinations
for OS in "${OS_values[@]}"; do
    for ORDER in "${ORDER_values[@]}"; do
        for INTEGRATOR in "${INTEGRATOR_values[@]}"; do

            TEST_TYPE="${OS}_${ORDER}_${INTEGRATOR}"
            final_output_dir="${OUTPUT_DIR}/${TEST_TYPE}"
            mkdir -p $final_output_dir

            echo "Running combination: $output_file"
            echo "
            OperatorSplitting = $OS
            Order = $ORDER
            Integrator = $INTEGRATOR
            "

            python train_swe.py \
                model.os=$OS \
                model.order=$ORDER \
                model.integrator=$INTEGRATOR \
                output_dir=$final_output_dir\
                > "$final_output_dir/log.txt"

        done
    done
done
