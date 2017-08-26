#!/bin/sh

export CUDA_VISIBLE_DEVICES=2
cd tutorial
python cifar10_train_adv.py --data_dir=.. --train_dir=train_adv_16 --max_steps=100000 --log_frequency=100 --epsilon=16
