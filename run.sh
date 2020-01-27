#!/bin/bash

#GPU=2
#screen -dmS b bash -c "export CUDA_VISIBLE_DEVICES=$GPU;python bird.py --dataset gtzan --option morlet -LR 0.001"
#GPU=6
#screen -dmS b bash -c "export CUDA_VISIBLE_DEVICES=$GPU;python bird.py --dataset gtzan --option morlet -LR 0.005"
#GPU=0
#screen -dmS d bash -c "export CUDA_VISIBLE_DEVICES=$GPU;python bird.py --dataset bird --option wvd -L 8 -LR 0.005 --run 1"
GPU=0
for run in 0
do
    for LR in 0.001
    do
#        GPU=$(((GPU+1)%8))
        screen -dmS d bash -c "export CUDA_VISIBLE_DEVICES=$GPU;python -i bird.py --dataset dyni --option melspec --bins 1024 -L 4 -LR $LR --model small --run $run  -BS 16 --hop 128"
    done
done    
