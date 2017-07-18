#!/bin/sh

export CUDA_VISIBLE_DEVICES=4
python resnet_main_ensadv.py --train_data_path=cifar-10-batches-bin/data_batch* --log_root=log_ensadv --train_dir=log_ensadv/train --dataset=cifar10 --num_gpus=1
