#!/bin/sh

export CUDA_VISIBLE_DEVICES=5
cd tutorial
python cifar10_train.py --data_dir=.. --train_dir=train_keep --max_steps=100000 --log_frequency=100
