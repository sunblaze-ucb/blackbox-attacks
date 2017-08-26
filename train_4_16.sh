#!/bin/sh

export CUDA_VISIBLE_DEVICES=0
python resnet_main_adv.py --train_data_path=cifar-10-batches-bin/data_batch* --log_root=log_adv_16 --train_dir=log_adv_16/train --dataset=cifar10 --num_gpus=1 --epsilon=16
