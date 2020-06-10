#!/bin/bash

GPUS=(5 6 7 4 0 1 2 3 4 0 1 2 3 4 0 1 2 0 1 2 2 1 2 1 5 6 7 0 1 4 0)
i=0
dataset=fsd
J=5
Q=16
hop=256
bins=1024
bs=16
#onelayer_linear_scattering onelayer_nonlinear_scattering joint_l 
#                                                inear_scattering
# 0.005 0.001 0.0002
for LR in 0.0002 0.001 0.005
do
    echo $LR
    for model in onelayer_linear_scattering onelayer_nonlinear_scattering
    do
        for option in learnmorlet
        do
	    echo $option
            GPU=${GPUS[i]}
    	    i=$((i+1))
            screen -dmS mnist$dataset$LR$option$model bash -c "export CUDA_VISIBLE_DEVICES=$GPU;python dyni.py --dataset $dataset --option $option --bins $bins -LR $LR --model $model -BS $bs --hop $hop -J $J -Q $Q"
        done
#        GPU=${GPUS[i]}
#        screen -dmS mnistwvd1$dataset$LR$model bash -c "export CUDA_VISIBLE_DEVICES=$GPU;python dyni.py --dataset $dataset --option wvd --bins $bins -L 6 -LR $LR --model $model -BS $bs --hop $hop -J $J -Q $Q"
#        i=$((i+1))
    done
done
