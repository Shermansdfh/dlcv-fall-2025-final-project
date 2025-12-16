#!/bin/bash
#PBS -N cld_format_pbs
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o $PBS_O_WORKDIR/hpc/logs/cld_format.$JOB_ID.log

# ==========================================
cd $PBS_O_WORKDIR

# export UV_OFFLINE=1
# export HF_DATASETS_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1

echo "Starting job on $(hostname)"

# Set PyTorch memory allocator to use expandable segments to reduce fragmentation
export PYTORCH_ALLOC_CONF=expandable_segments:True

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate CLD
python src/adapters/rtdetr_layerd_to_cld_infer.py --config configs/exp001/pipeline.yaml
conda deactivate

echo "Job finished"