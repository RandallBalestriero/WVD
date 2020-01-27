#!/bin/bash

#GPU=2
#screen -dmS b bash -c "export CUDA_VISIBLE_DEVICES=$GPU;python bird.py --dataset gtzan --option morlet -LR 0.001"
#GPU=6
#screen -dmS b bash -c "export CUDA_VISIBLE_DEVICES=$GPU;python bird.py --dataset gtzan --option morlet -LR 0.005"
#GPU=0
#screen -dmS d bash -c "export CUDA_VISIBLE_DEVICES=$GPU;python bird.py --dataset bird --option wvd -L 8 -LR 0.005 --run 1"
GPU=0
BINS=1024
HOP=128
BS=16
MODEL=small

for run in 0
do
    for LR in 0.001
    do
        GPU=$(((GPU+1)%8))
        screen -dmS d bash -c "export CUDA_VISIBLE_DEVICES=$GPU;python dyni.py --option melspec --bins $BINS -LR $LR --model $MODEL --run $run  -BS $BS --hop $HOP"
        GPU=$(((GPU+1)%8))
        screen -dmS d bash -c "export CUDA_VISIBLE_DEVICES=$GPU;python dyni.py --option wvd --bins $BINS -LR $LR --model $MODEL --run $run  -BS $BS --hop $HOP"
        GPU=$(((GPU+1)%8))
        screen -dmS d bash -c "export CUDA_VISIBLE_DEVICES=$GPU;python dyni.py --option morlet --bins $BINS -LR $LR --model $MODEL --run $run  -BS $BS --hop $HOP"
        GPU=$(((GPU+1)%8))
        screen -dmS d bash -c "export CUDA_VISIBLE_DEVICES=$GPU;python dyni.py --option sinc --bins $BINS -LR $LR --model $MODEL --run $run  -BS $BS --hop $HOP"
    done
done    
