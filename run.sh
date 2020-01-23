#!/bin/bash

#screen -dmS a bash -c "export CUDA_VISIBLE_DEVICES=0;python bird.py -option sinc -BS 5 -J 40";
#screen -dmS b bash -c "export CUDA_VISIBLE_DEVICES=1;python bird.py -option melspec -BS 5 -J 40";
#screen -dmS a bash -c "export CUDA_VISIBLE_DEVICES=0;python bird.py -option wvd8 -BS 5 -J 40 -model base";
#screen -dmS b bash -c "export CUDA_VISIBLE_DEVICES=1;python bird.py -option wvd8 -BS 5 -J 40 -model small";
export CUDA_VISIBLE_DEVICES=''
#nohup python bird.py --dataset gtzan --option melspec -L 2 -LR 0.0005 > file1.output &
#nohup python bird.py --dataset gtzan --option morlet -L 2 -LR 0.005 > file2.output &
nohup python bird.py --dataset gtzan --option wvd -L 2 -LR 0.005 > file3.output &
nohup python bird.py --dataset gtzan --option wvd -L 8 -LR 0.005 > file4.output &




#
