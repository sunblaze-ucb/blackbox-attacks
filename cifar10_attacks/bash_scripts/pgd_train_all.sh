#!/bin/sh -ex

export CUDA_VISIBLE_DEVICES=1

#python baseline_cifar10.py thin_32_pgd

#python baseline_cifar10.py thin_32_pgd --alpha=33

python cifar10_query_based.py thin_32_pgd

python cifar10_query_based.py thin_32_pgd --loss_type=cw