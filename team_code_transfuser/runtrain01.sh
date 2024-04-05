#!/bin/bash -l

# specifying arguments to the batch job. Its easier to enter it here
#$ -P rlvn
#$ -N transfusertraining
#$ -o TT_output
#$ -e TT_error
#$ -m b
#$ -l gpus=1

# module loading
ml miniconda
conda activate /projectnb/rlvn/students/vrs/anaconda3/envs/tfuse

# running the training
python /projectnb/rlvn/students/vrs/transfuser/team_code_transfuser/train01.py --batch_size 10 --logdir /projectnb/rlvn/students/vrs/transfuser/logdirectory --root_dir /projectnb/rlvn/students/vrs/transfuser/data/ --parallel_training 0