#!/bin/bash

#GPU=2
#screen -dmS b bash -c "export CUDA_VISIBLE_DEVICES=$GPU;python bird.py --dataset gtzan --option morlet -LR 0.001"
#GPU=6
#screen -dmS b bash -c "export CUDA_VISIBLE_DEVICES=$GPU;python bird.py --dataset gtzan --option morlet -LR 0.005"
#GPU=0
#screen -dmS d bash -c "export CUDA_VISIBLE_DEVICES=$GPU;python bird.py --dataset bird --option wvd -L 8 -LR 0.005 --run 1"

for run in 0 1
do
    for LR in 0.001 0.005
    do
    #    GPU=$(((GPU+1)%8))
    #    screen -dmS a bash -c "export CUDA_VISIBLE_DEVICES=$GPU;python bird.py --dataset gtzan --option melspec -LR $LR"
    #    GPU=$(((GPU+1)%8))
    #    screen -dmS b bash -c "export CUDA_VISIBLE_DEVICES=$GPU;python bird.py --dataset gtzan --option morlet -LR $LR"
    #    GPU=$(((GPU+1)%8))
    #    screen -dmS c bash -c "export CUDA_VISIBLE_DEVICES=$GPU;python bird.py --dataset gtzan --option sinc -LR $LR"
        GPU=$(((GPU+1)%8))
        screen -dmS d bash -c "export CUDA_VISIBLE_DEVICES=$GPU;python bird.py --dataset gtzan --option wvd -L 8 -LR $LR --run $run"
        GPU=$(((GPU+1)%8))
        screen -dmS d bash -c "export CUDA_VISIBLE_DEVICES=$GPU;python bird.py --dataset bird --option wvd -L 8 -LR $LR --run $run"
    done
done    
