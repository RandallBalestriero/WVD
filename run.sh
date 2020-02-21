#!/bin/bash

GPUS=(5 6 7 6 2 3 4 5 6 7 2 3 4 5 6 7 2 3 4 5 6 7)
i=0
dataset=esc
J=5
Q=16
hop=512
bins=1024
bs=16
#onelayer_linear_scattering onelayer_nonlinear_scattering joint_linear_scattering
for LR in 0.0002
do
    for model in joint_nonlinear_scattering
    do
#        for option in sinc learnmorlet
#        do
#            GPU=${GPUS[i]}
#    	    i=$((i+1))
#            screen -dmS mnist$LR$option bash -c "export CUDA_VISIBLE_DEVICES=$GPU;python -i dyni.py --dataset $dataset --option $option --bins $bins -LR $LR --model $model -BS $bs --hop $hop -J $J -Q $Q"
#        done
        GPU=${GPUS[i]}
        i=$((i+1))
        screen -dmS mnistwvd1$dataset$LR bash -c "export CUDA_VISIBLE_DEVICES=$GPU;python dyni.py --dataset $dataset --option wvd --bins $bins -L 6 -LR $LR --model $model -BS $bs --hop $hop -J $J -Q $Q"
        GPU=${GPUS[i]}
        i=$((i+1))
        screen -dmS mnistwvd2$dataset$LR bash -c "export CUDA_VISIBLE_DEVICES=$GPU;python dyni.py --dataset $dataset --option mwvd --bins $bins -L 6 -LR $LR --model $model -BS $bs --hop $hop -J $J -Q $Q" 
    done
done
