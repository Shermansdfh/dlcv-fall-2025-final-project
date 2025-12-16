#!/bin/bash
#PBS -N full_pipeline
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o pbs_full_pipeline_out.log

# ==========================================
cd $PBS_O_WORKDIR

# export UV_OFFLINE=1
# export HF_DATASETS_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1

echo "Starting job on $(hostname)"
python -u run_full_pipeline.py --config pipeline_config.yaml > full_pipeline.log 2>&1
echo "Job finished"