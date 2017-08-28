#!/bin/sh -ex

export CUDA_VISIBLE_DEVICES=4

test -e test_random_targets.npy

python generate_test_opt_targeted.py test_thin_opt_targeted_8.npy log 8 6705
