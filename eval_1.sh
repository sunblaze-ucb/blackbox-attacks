#!/bin/sh

export CUDA_VISIBLE_DEVICES=4
python eval_static_distorted.py --train_data_path=cifar-10-batches-bin/data_batch* --log_root=log
