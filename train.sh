#!/bin/bash
OS="strang"
ORDER="ADR"
INTEGRATOR="FE"


output_dir="/home/ndj376/ADRnet/AdvectionNet/PDEBench_SWE_ADRNet_Pred50/test_results/"
mkdir -p $output_dir
output_file=${OS}_${ORDER}_${INTEGRATOR}


echo $output_file 
echo "
OperatorSplitting= $OS
Order=$ORDER
Integrator=$INTEGRATOR
"

python train_swe.py \
    model.os=$OS \
    model.order=$ORDER \
    model.integrator=$INTEGRATOR \
    hydra.run.dir=${output_dir} \
    hydra.output_subdir=null \
    > ${output_dir}/${output_file}.txt