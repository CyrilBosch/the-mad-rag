# Initialize conda
source $(conda info --base)/etc/profile.d/conda.sh

# Activate the conda environment
conda activate mad-rag-env

python test.py