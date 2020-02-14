#!/bin/bash

#GPU=2
#screen -dmS b bash -c "export CUDA_VISIBLE_DEVICES=$GPU;python bird.py --dataset gtzan --option morlet -LR 0.001"
#GPU=6
#screen -dmS b bash -c "export CUDA_VISIBLE_DEVICES=$GPU;python bird.py --dataset gtzan --option morlet -LR 0.005"
#GPU=0
#screen -dmS d bash -c "export CUDA_VISIBLE_DEVICES=$GPU;python bird.py --dataset bird --option wvd -L 8 -LR 0.005 --run 1"
GPUS=(4 5 6 7 4 5 4 5 6 7 6 7)
i=0
dataset=mnist

for LR in 0.0002
do
    for model in scattering small
    do
        for option in learnmorlet sinc melspec morlet
        do
            GPU=${GPUS[i]}
    	    i=$((i+1))
            screen -dmS mnist$LR$option bash -c "export CUDA_VISIBLE_DEVICES=$GPU;python -i dyni.py --dataset tut --option $option --bins 1024 -LR $LR --model $model -BS 10 --hop 512"
        done
#        GPU=${GPUS[i]}
#   	    i=$((i+1))
#        screen -dmS mnistwvd1$LR bash -c "export CUDA_VISIBLE_DEVICES=$GPU;python dyni.py --dataset gtzan --option $option --bins 1024 -L 4 -LR $LR --model $model -BS 10 --hop 512 --modes 1"
#        GPU=${GPUS[i]}
# 	    i=$((i+1))
#        screen -dmS mnistwvd2$LR bash -c "export CUDA_VISIBLE_DEVICES=$GPU;python dyni.py --dataset gtzan --option $option --bins 1024 -L 4 -LR $LR --model $model -BS 10 --hop 512 --modes 3" 
    done
done
