#!/bin/bash

screen -dmS a bash -c "export CUDA_VISIBLE_DEVICES=0;python bird.py -option sinc -BS 5 -J 40";
screen -dmS b bash -c "export CUDA_VISIBLE_DEVICES=1;python bird.py -option melspec -BS 5 -J 40";
screen -dmS c bash -c "export CUDA_VISIBLE_DEVICES=2;python bird.py -option wvd8 -BS 5 -J 40";




