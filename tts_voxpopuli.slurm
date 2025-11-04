#!/bin/sh
#SBATCH --account=studiegrupper-cogito
#SBATCH --job-name="tts_voxpopuli"
#SBATCH --time=20:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --nodes=1
#SBATCH --output=slurm_outputs/output_pytorch_tts_voxpopuli.txt
#SBATCH --error=slurm_outputs/output_pytorch_tts_voxpopuli.err
#SBATCH --mail-user=nathanim@ntnu.no
#SBATCH --mail-type=ALL


WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"

ENV_PATH="/cluster/work/$(whoami)/Audify/conda-env/"

module purge
module load Anaconda3/2024.02-1

# Create a new Conda environment
conda create -y --prefix ${ENV_PATH} python=3.11

# Activate the environment
conda activate ${ENV_PATH}

echo "Downloaded dependencies:"
#pip install transformers "datasets==3.6.0" soundfile speechbrain accelerate
pip install torch==2.0.1 torchaudio==2.0.2 speechbrain==0.5.15 transformers datasets==3.6.0 soundfile accelerate wandb python-dotenv librosa matplotlib
pip freeze



# Run the Python script
python ./src/tts_voxpopuli.py

# Deactivate the environment
conda deactivate

echo "Job finished"

# "sbatch ./tts_voxpopuli.slurm"  to run the job