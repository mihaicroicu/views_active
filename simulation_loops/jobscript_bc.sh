#!/usr/bin/env bash
#SBATCH -A SNIC2022-5-587 -p alvis
#SBATCH -N 1 --gpus-per-node=A40:1  
#SBATCH -t 0-11:00:00

input_file=$1  # opting to also take the file as an input argument

# Read the given line from the input file and evaluate it:
eval `head -n $SLURM_ARRAY_TASK_ID $input_file | tail -1`

module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0 SciPy-bundle/2022.05-foss-2022a  IPython scikit-learn/1.1.2-foss-2022a matplotlib/3.5.2-foss-2022a
source /cephyr/users/croicu/Alvis/marisol/bin/activate

echo "[`date`] Running active_learner=$active_learner adam_lr=$adam_lr seed=$seed"

python experiment_marisol_bc.py --seed=$seed --adam_lr=$adam_lr --active_learner=$active_learner
