#!/bin/bash

#GPU=2
#screen -dmS b bash -c "export CUDA_VISIBLE_DEVICES=$GPU;python bird.py --dataset gtzan --option morlet -LR 0.001"
#GPU=6
#screen -dmS b bash -c "export CUDA_VISIBLE_DEVICES=$GPU;python bird.py --dataset gtzan --option morlet -LR 0.005"
#GPU=0
#screen -dmS d bash -c "export CUDA_VISIBLE_DEVICES=$GPU;python bird.py --dataset bird --option wvd -L 8 -LR 0.005 --run 1"
GPUS=(0 1 2 1 2 0 2 0 1)
i=0

for run in 0
do
    for model in small scattering
    do
        for option in wvd learnmorlet sinc
        do
            for LR in 0.0002
            do
                GPU=${GPUS[i]}
    	    i=$((i+1))
                screen -dmS mnist$LR$option bash -c "export CUDA_VISIBLE_DEVICES=$GPU;python dyni.py --dataset mnist --option $option --bins 1024 -L 4 -LR $LR --model $model --run $run  -BS 10 --hop 128"
            done
        done
    done

#    for option in wvd morlet sinc
#    do
#        for LR in 0.001 0.005 0.0002
#        do
#            GPU=${GPUS[i]}
#	    i=$((i+1))
#            screen -dmS dyni$LR$option bash -c "export CUDA_VISIBLE_DEVICES=$GPU;python dyni.py --dataset dyni --option $option --bins 1024 -L 4 -LR $LR --model small --run $run  -BS 10 --hop 128"
#        done
#    done

done
