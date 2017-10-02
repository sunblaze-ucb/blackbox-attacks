#!/bin/sh -ex

export CUDA_VISIBLE_DEVICES=2

python cifar10_query_based.py tutorial --delta=1.0 --loss_type=xent
#python cifar10_query_based.py tutorial --loss_type=cw
python cifar10_query_based.py tutorial_adv --delta=1.0 --loss_type=xent
python cifar10_query_based.py tutorial_adv --loss_type=cw
python cifar10_query_based.py tutorial_ensadv --delta=1.0 --loss_type=xent
python cifar10_query_based.py tutorial_ensadv --loss_type=cw
