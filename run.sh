#!/bin/bash


#screen -dmS a bash -c "export CUDA_VISIBLE_DEVICES='1';python bird.py --dataset gtzan --option melspec -LR 0.001"

#screen -dmS b bash -c "export CUDA_VISIBLE_DEVICES='1';python bird.py --dataset gtzan --option morlet -LR 0.001"

screen -dmS b bash -c "export CUDA_VISIBLE_DEVICES='1';python bird.py --dataset gtzan --option sinc -LR 0.001"

#screen -dmS c bash -c "export CUDA_VISIBLE_DEVICES='2';python bird.py --dataset gtzan --option wvd -L 8 -LR 0.001"

