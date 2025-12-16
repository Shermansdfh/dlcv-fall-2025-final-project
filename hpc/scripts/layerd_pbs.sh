#!/bin/bash
#PBS -N decomopose_pbs
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o $PBS_O_WORKDIR/hpc/logs/decompose.$JOB_ID.log

# ==========================================
cd $PBS_O_WORKDIR

# export UV_OFFLINE=1
# export HF_DATASETS_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1

echo "Starting job on $(hostname)"

# Set PyTorch memory allocator to use expandable segments to reduce fragmentation
export PYTORCH_ALLOC_CONF=expandable_segments:True

uv run python src/layerd/infer.py --config configs/exp001/pipeline.yaml > experiments/exp001/ld_to_pipeline_infer.log 2>&1
echo "Job finished"