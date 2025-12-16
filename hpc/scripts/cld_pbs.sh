
#!/bin/bash
#PBS -N cld_run_pbs
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o $PBS_O_WORKDIR/hpc/logs/cld_run.$JOB_ID.log   

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
python src/cld/infer_dlcv.py --config_path configs/exp001/cld/infer.yaml > experiments/exp001/cld_infer.log 2>&1
conda deactivate
echo "Job finished"