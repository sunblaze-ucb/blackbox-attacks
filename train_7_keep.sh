#!/bin/sh

export CUDA_VISIBLE_DEVICES=4
python resnet_main_ensadv.py --train_data_path=cifar-10-batches-bin/data_batch* --log_root=log_ensadv_keep --train_dir=log_ensadv_keep/train --dataset=cifar10 --num_gpus=1
