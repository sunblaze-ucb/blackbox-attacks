#!/bin/sh -ex

export CUDA_VISIBLE_DEVICES=3

python cifar10_query_based.py thin_32 --method=query_based_iter --loss_type=xent --delta=1.0 &

export CUDA_VISIBLE_DEVICES=4

python cifar10_query_based.py thin_32 --method=query_based_iter --loss_type=cw &

export CUDA_VISIBLE_DEVICES=5

python cifar10_query_based.py wide_28_10 --method=query_based_iter --loss_type=xent --delta=1.0 &

export CUDA_VISIBLE_DEVICES=6

python cifar10_query_based.py wide_28_10 --method=query_based_iter --loss_type=cw &

export CUDA_VISIBLE_DEVICES=7

python cifar10_query_based.py tutorial --method=query_based_iter --loss_type=xent --delta=1.0 &

export CUDA_VISIBLE_DEVICES=2

python cifar10_query_based.py tutorial --method=query_based_iter --loss_type=cw &
