#!/bin/bash
##SBATCH --ntasks=28
#SBATCH --job-name=trp_1
#SBATCH --time=24:0:0
#SBATCH -N 1
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks-per-core=1
##SBATCH --mem=200g
# load necessary module

#module load cuDNN/7.6.5/CUDA-10.1
module laod cuda/10.2

pwd=$PWD
source ~/.bashrc
cd $pwd

conda activate koopnet

for i in {1..10};do
python train.py --epochs 100 --batch-size 1000 --num-atoms 20 --num-classes 5 --save-folder logs_$i --h_a 16 --num_neighbors 7 --n_conv 4 --h_g 2 --conv_type SchNet --dmin 0. --dmax 8. --step 0.5 --tau 5 --train --dist-data ../dists_trpcage_ca_7nbrs_1ns.npz --nbr-data ../inds_trpcage_ca_7nbrs_1ns.npz --residual
done
