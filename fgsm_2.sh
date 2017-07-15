#!/bin/sh

export CUDA_VISIBLE_DEVICES=7
python generate_fgsm_wide.py --train_data_path=cifar-10-batches-bin/data_batch* --log_root=log_wide
