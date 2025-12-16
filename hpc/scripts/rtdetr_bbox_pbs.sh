#!/bin/bash
#PBS -N detection_pbs
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o $PBS_O_WORKDIR/hpc/logs/detection.$JOB_ID.log

# ==========================================
cd $PBS_O_WORKDIR

# export UV_OFFLINE=1
# export HF_DATASETS_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1

echo "Starting job on $(hostname)"

uv run python src/bbox/infer.py --config configs/exp001/pipeline.yaml
echo "Job finished"