#!/bin/sh -ex

export CUDA_VISIBLE_DEVICES=1

python cifar10_query_based.py wide_28_10 --delta=1.0 --loss_type=xent
python cifar10_query_based.py wide_28_10 --loss_type=cw
python cifar10_query_based.py wide_28_10_adv --delta=1.0 --loss_type=xent
python cifar10_query_based.py wide_28_10_adv --loss_type=cw
python cifar10_query_based.py wide_28_10_ensadv --delta=1.0 --loss_type=xent
python cifar10_query_based.py wide_28_10_ensadv --loss_type=cw
