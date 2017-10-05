#!/bin/sh

export CUDA_VISIBLE_DEVICES=3
python resnet_main_ensadv_3210.py --train_data_path=cifar-10-batches-bin/data_batch* --log_root=log_ensadv_3210 --train_dir=log_ensadv_3210/train --dataset=cifar10 --num_gpus=1
