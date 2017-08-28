#!/bin/sh -ex

export CUDA_VISIBLE_DEVICES=5
export TF_CPP_MIN_LOG_LEVEL=2

test -e test_random_targets.npy || python generate_test_random_targets.py

python generate_test_fgsm_targeted.py test_thin_fgsm_targeted_8.npy log 8
python generate_test_fgsmlogit_targeted.py test_thin_fgsmlogit_targeted_8.npy log 8

# python generate_test_fgsm_targeted.py test_thin_fgsm_targeted_16.npy log 16
# python generate_test_fgsmlogit_targeted.py test_thin_fgsmlogit_targeted_16.npy log 16
