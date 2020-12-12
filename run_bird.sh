#!/bin/bash

GPUS=(0 1 2 3 4 5 6 0 0 1 2 0 0 0 3 4 5 6 6 5 6 3 4 5 6 5 6 7 4 0 1 2 3 4 0 1 2 3 4 0 1 2 0 1 2 2 1 2 1 5 6 7 0 1 4 0)
i=0
dataset=commands
J=4
Q=16
bins=512
bs=10
#onelayer_linear_scattering onelayer_nonlinear_scattering joint_l 
#                                                inear_scattering
# 0.005 0.001 0.0002
for LR in 0.0002 0.001 0.005
do
    echo $LR
    for model in deep_net
    do
        # for option in morlet learnmorlet sinc
        # do
	       # echo $option
        #     GPU=${GPUS[i]}
    	   #  i=$((i+1))
        #     screen -dmS bird$dataset$LR$option bash -c "export CUDA_VISIBLE_DEVICES=$GPU;python -i dyni.py --dataset $dataset --option $option -lr $LR --model $model -BS $bs -J $J -Q $Q --epochs 50"
        # done

        GPU=${GPUS[i]}
        i=$((i+1))
        screen -dmS birdwvd$dataset$LR bash -c "export CUDA_VISIBLE_DEVICES=$GPU;python -i dyni.py --dataset $dataset --option wvd -L 6 -lr $LR --model $model -BS $bs -J $J -Q $Q --epochs 50 --wvdinit gabor"
        # GPU=${GPUS[i]}
        # i=$((i+1))
        # screen -dmS birdstftwvd$dataset$LR bash -c "export CUDA_VISIBLE_DEVICES=$GPU;python dyni.py --dataset $dataset --option wvd -L 8 -lr $LR --model $model -BS $bs -J $J -Q $Q --epochs 50 --wvdinit stftsmall"
        # GPU=${GPUS[i]}
        # i=$((i+1))
        # screen -dmS birdwvd$dataset$LR bash -c "export CUDA_VISIBLE_DEVICES=$GPU;python dyni.py --dataset $dataset --option wvd -L 8 -lr $LR --model $model -BS $bs -J $J -Q $Q --epochs 50 --wvdinit stftsmall"
    done
done
