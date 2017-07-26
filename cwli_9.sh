#!/bin/sh -ex

export CUDA_VISIBLE_DEVICES=6
python generate_test_cwli.py test_tutorial_cwli_16.npy tutorial/train 16 6705
