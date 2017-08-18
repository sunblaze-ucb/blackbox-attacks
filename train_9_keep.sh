#!/bin/sh

export CUDA_VISIBLE_DEVICES=7
cd tutorial
python cifar10_train_ensadv.py --data_dir=.. --train_dir=train_ensadv_keep --max_steps=100000 --log_frequency=100
