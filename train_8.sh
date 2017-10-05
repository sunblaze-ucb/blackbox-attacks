#!/bin/sh

export CUDA_VISIBLE_DEVICES=5
python resnet_main_ensadv_wide.py --train_data_path=cifar-10-batches-bin/data_batch* --log_root=log_ensadv_wide --train_dir=log_ensadv_wide/train --dataset=cifar10 --num_gpus=1
