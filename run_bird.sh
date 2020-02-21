#!/bin/bash

GPUS=(5 5 5 5 3 4 3 4 0 1 2 3 4 0 1 2 3 4 0 1 2 3 4)
i=0
dataset=bird
J=5
Q=16
hop=64
bins=1024
bs=16

for LR in 0.0002
do
    for model in onelayer_linear_scattering onelayer_nonlinear_scattering joint_linear_scattering joint_nonlinear_scattering
    do
        for option in sinc learnmorlet
        do
            GPU=${GPUS[i]}
    	    i=$((i+1))
            screen -dmS mnist$dataset$LR$option bash -c "export CUDA_VISIBLE_DEVICES=$GPU;python dyni.py --dataset $dataset --option $option --bins $bins -LR $LR --model $model -BS $bs --hop $hop -J $J -Q $Q"
        done
#        GPU=${GPUS[i]}
#        i=$((i+1))
#        screen -dmS mnistwvd1$dataset$LR bash -c "export CUDA_VISIBLE_DEVICES=$GPU;python dyni.py --dataset $dataset --option wvd --bins $bins -L 6 -LR $LR --model $model -BS $bs --hop $hop -J $J -Q $Q"
#        GPU=${GPUS[i]}
#        i=$((i+1))
#        screen -dmS mnistwvd2$dataset$LR bash -c "export CUDA_VISIBLE_DEVICES=$GPU;python dyni.py --dataset $dataset --option mwvd --bins $bins -L 6 -LR $LR --model $model -BS $bs --hop $hop -J $J -Q $Q" 
    done
done
