#!/bin/bash

GPUS=(0 3 4 6 7 0 3 4 6 7 0 2 4 6 7 5 1 2 3 0 1 2 4 0 1 2 3 5 4 5 0 1 0 1 0 1 4 5 6 7 2 3 4 0 1 2 3 4 5 6 0 0 1 2 0 0 0 3 4 5 6 6 5 6 3 4 5 6 5 6 7 4 0 1 2 3 4 0 1 2 3 4 0 1 2 0 1 2 2 1 2 1 5 6 7 0 1 4 0)
i=0
dataset=quebec
J=5
Q=8
bins=512
bs=16
#onelayer_linear_scattering onelayer_nonlinear_scattering joint_l 
#                                                inear_scattering
# 0.005 0.001 0.0002
for LR in 0.001 0.005 0.0002
do
    echo $LR
    for model in onelayer_linear_scattering
    do
       for option in sinc morlet
       do
	       echo $option
           GPU=${GPUS[i]}
   	    i=$((i+1))
           screen -dmS bird$dataset$LR$option bash -c "export CUDA_VISIBLE_DEVICES=$GPU;python -i run.py --dataset $dataset --option $option -lr $LR --model $model -BS $bs -J $J -Q $Q --epochs 50"
       done

        # GPU=${GPUS[i]}
        # i=$((i+1))
        # screen -dmS birdwvd$dataset$LR bash -c "export CUDA_VISIBLE_DEVICES=$GPU;python -i run.py --dataset $dataset --option wvd -L 8 -lr $LR --model $model -BS $bs -J $J -Q $Q --epochs 50 --wvdinit gabor"
        # GPU=${GPUS[i]}
        # i=$((i+1))
        # screen -dmS birdstftwvd$dataset$LR bash -c "export CUDA_VISIBLE_DEVICES=$GPU;python run.py --dataset $dataset --option wvd -L 8 -lr $LR --model $model -BS $bs -J $J -Q $Q --epochs 50 --wvdinit stftsmall"
        # GPU=${GPUS[i]}
        # i=$((i+1))
        # screen -dmS birdwvd$dataset$LR bash -c "export CUDA_VISIBLE_DEVICES=$GPU;python run.py --dataset $dataset --option wvd -L 8 -lr $LR --model $model -BS $bs -J $J -Q $Q --epochs 50 --wvdinit stftsmall"
    done
done
