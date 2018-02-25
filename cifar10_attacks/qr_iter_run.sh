#!/bin/sh -ex

export CUDA_VISIBLE_DEVICES=3

python cifar10_query_based.py thin_32 --method=query_based_iter --num_comp=400

python cifar10_query_based.py thin_32 --method=query_based_iter --group_size=8

python cifar10_query_based.py wide_28_10 --method=query_based_iter --num_comp=400

python cifar10_query_based.py wide_28_10 --method=query_based_iter --group_size=8