#!/bin/sh

export CUDA_VISIBLE_DEVICES=7
cd tutorial
python generate_fgsm.py --data_dir=.. --train_dir=train --batch_size=100
