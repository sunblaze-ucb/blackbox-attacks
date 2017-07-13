#!/bin/sh

export CUDA_VISIBLE_DEVICES=4
python resnet_main.py --train_data_path=cifar-10-batches-bin/data_batch* --log_root=log --train_dir=log/train --dataset=cifar10 --num_gpus=1
