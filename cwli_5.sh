#!/bin/sh -ex

export CUDA_VISIBLE_DEVICES=2
python generate_test_cwli.py test_wide_cwli_12.npy log_wide 12 6705
