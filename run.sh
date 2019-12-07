#!/bin/bash

screen -dmS a bash -c "export CUDA_VISIBLE_DEVICES=0;python bird.py -L 0";
screen -dmS b bash -c "export CUDA_VISIBLE_DEVICES=1;python bird.py -L 8";
screen -dmS c bash -c "export CUDA_VISIBLE_DEVICES=2;python bird.py -L 16";




