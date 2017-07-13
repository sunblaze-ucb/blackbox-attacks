#!/bin/sh

export CUDA_VISIBLE_DEVICES=7
python resnet_main_adv.py --train_data_path=cifar-10-batches-bin/data_batch* --log_root=log_adv --train_dir=log_adv/train --dataset=cifar10 --num_gpus=1
