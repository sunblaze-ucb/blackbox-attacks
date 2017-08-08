#!/bin/sh -ex

export CUDA_VISIBLE_DEVICES=7

python baseline_cifar10.py thin_32 --alpha=33.0
python baseline_cifar10.py thin_32_adv --alpha=33.0
python baseline_cifar10.py thin_32_ensadv --alpha=33.0
python baseline_cifar10.py wide_28_10 --alpha=33.0
python baseline_cifar10.py wide_28_10_adv --alpha=33.0
python baseline_cifar10.py wide_28_10_ensadv --alpha=33.0
python baseline_cifar10.py tutorial --alpha=33.0
python baseline_cifar10.py tutorial_adv --alpha=33.0
python baseline_cifar10.py tutorial_ensadv --alpha=33.0