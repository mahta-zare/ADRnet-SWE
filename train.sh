#!/bin/bash
OS="lie"
ORDER="ADR"
INTEGRATOR="FE"
EPOCHS=2
CUDA="0"
OUTPUT_DIR="/home/ndj376/ADRnet/AdvectionNet/PDEBench_SWE_ADRNet_Pred50/test_results"



TEST_TYPE="${OS}_${ORDER}_${INTEGRATOR}"
timestamp=$(date +"%Y%m%d_%H%M%S")
FINAL_OUTPUT_DIR="${OUTPUT_DIR}/${TEST_TYPE}_${timestamp}"
mkdir -p $FINAL_OUTPUT_DIR

echo "Output directory: $FINAL_OUTPUT_DIR"

echo "
OperatorSplitting=$OS
Order=$ORDER
Integrator=$INTEGRATOR
EPOCHS=$EPOCHS
" > $FINAL_OUTPUT_DIR/log.txt


python train_swe.py \
    model.os=$OS \
    model.order=$ORDER \
    model.integrator=$INTEGRATOR \
    output_dir=$FINAL_OUTPUT_DIR \
    compute.cuda_visible_devices=$CUDA \
    trainer.num_epochs=$EPOCHS \
    >> $FINAL_OUTPUT_DIR/log.txt