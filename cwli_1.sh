#!/bin/sh -ex

export CUDA_VISIBLE_DEVICES=4
python generate_test_cwli.py test_thin_cwli_8.npy log 8 6705
