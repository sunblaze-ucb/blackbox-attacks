#!/bin/sh

export CUDA_VISIBLE_DEVICES=5
cd tutorial
python cifar10_train_adv.py --data_dir=.. --train_dir=train_adv
