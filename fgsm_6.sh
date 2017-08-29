#!/bin/sh -ex

export CUDA_VISIBLE_DEVICES=5
export TF_CPP_MIN_LOG_LEVEL=2

test -e test_random_targets.npy || python generate_test_random_targets.py

### targeted attacks

# python generate_test_fgsm_targeted.py test_thin_fgsm_targeted_8.npy log 8
# python generate_test_fgsmlogit_targeted.py test_thin_fgsmlogit_targeted_8.npy log 8
# python generate_test_itergs_targeted.py test_thin_itergs_targeted_8.npy log 8 1 10
# python generate_test_itergslogit_targeted.py test_thin_itergslogit_targeted_8.npy log 8 1 10

# python generate_test_fgsm_targeted.py test_thin_fgsm_targeted_16.npy log 16
# python generate_test_fgsmlogit_targeted.py test_thin_fgsmlogit_targeted_16.npy log 16

### untargeted attacks

# python generate_test_fgsmlogit.py test_thin_fgsmlogit_8.npy log 8
python generate_test_itergs.py test_thin_itergs_8.npy log 8 1 10
python generate_test_itergslogit.py test_thin_itergslogit_8.npy log 8 1 10
