#!/bin/sh

export CUDA_VISIBLE_DEVICES=3
python resnet_main_adv_wide.py --train_data_path=cifar-10-batches-bin/data_batch* --log_root=log_adv_wide_keep --train_dir=log_adv_wide_keep/train --dataset=cifar10 --num_gpus=1
