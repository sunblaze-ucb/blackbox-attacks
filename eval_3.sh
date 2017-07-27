#!/bin/sh -ex

export CUDA_VISIBLE_DEVICES=6

predict() { # src dst ckpt_dir
	python predict_file.py preds_$1_$2.npy test_$1.npy $3
}


predict_and_compare() { # src dst ckpt_dir [offset]
	predict $1 $2 $3
	python compare_preds.py preds_orig_$2.npy preds_$1_$2.npy $4
}

# predict orig thin log
# predict orig thin_adv log_adv
# predict orig thin_ensadv log_ensadv
# predict orig wide log_wide
# predict orig wide_adv log_adv_wide
# predict orig wide_ensadv log_ensadv_wide
# predict orig tutorial tutorial/train
# predict orig tutorial_adv tutorial/train_adv
# predict orig tutorial_ensadv tutorial/train_ensadv

# predict_and_compare wide_fgsm_8 thin_adv log_adv
# predict_and_compare wide_fgsm_8 thin_ensadv log_ensadv
# predict_and_compare tutorial_fgsm_8 wide_adv log_adv_wide
# predict_and_compare tutorial_fgsm_8 wide_ensadv log_ensadv_wide
# predict_and_compare thin_fgsm_8 tutorial_adv tutorial/train_adv
# predict_and_compare thin_fgsm_8 tutorial_ensadv tutorial/train_ensadv

# predict_and_compare wide_cwli_8 thin_adv log_adv 6705
# predict_and_compare wide_cwli_8 thin_ensadv log_ensadv 6705
# predict_and_compare tutorial_cwli_8 wide_adv log_adv_wide 6705
# predict_and_compare tutorial_cwli_8 wide_ensadv log_ensadv_wide 6705
# predict_and_compare thin_cwli_8 tutorial_adv tutorial/train_adv 6705
# predict_and_compare thin_cwli_8 tutorial_ensadv tutorial/train_ensadv 6705

# predict_and_compare wide_fgsm_12 thin_adv log_adv
# predict_and_compare wide_fgsm_12 thin_ensadv log_ensadv
# predict_and_compare tutorial_fgsm_12 wide_adv log_adv_wide
# predict_and_compare tutorial_fgsm_12 wide_ensadv log_ensadv_wide
# predict_and_compare thin_fgsm_12 tutorial_adv tutorial/train_adv
# predict_and_compare thin_fgsm_12 tutorial_ensadv tutorial/train_ensadv

# predict_and_compare wide_cwli_12 thin_adv log_adv 6705
# predict_and_compare wide_cwli_12 thin_ensadv log_ensadv 6705
# predict_and_compare tutorial_cwli_12 wide_adv log_adv_wide 6705
# predict_and_compare tutorial_cwli_12 wide_ensadv log_ensadv_wide 6705
# predict_and_compare thin_cwli_12 tutorial_adv tutorial/train_adv 6705
# predict_and_compare thin_cwli_12 tutorial_ensadv tutorial/train_ensadv 6705

# predict_and_compare wide_fgsm_16 thin_adv log_adv
# predict_and_compare wide_fgsm_16 thin_ensadv log_ensadv
# predict_and_compare tutorial_fgsm_16 wide_adv log_adv_wide
# predict_and_compare tutorial_fgsm_16 wide_ensadv log_ensadv_wide
# predict_and_compare thin_fgsm_16 tutorial_adv tutorial/train_adv
# predict_and_compare thin_fgsm_16 tutorial_ensadv tutorial/train_ensadv

# predict_and_compare wide_cwli_16 thin_adv log_adv 6705
# predict_and_compare wide_cwli_16 thin_ensadv log_ensadv 6705
# predict_and_compare tutorial_cwli_16 wide_adv log_adv_wide 6705
# predict_and_compare tutorial_cwli_16 wide_ensadv log_ensadv_wide 6705
# predict_and_compare thin_cwli_16 tutorial_adv tutorial/train_adv 6705
# predict_and_compare thin_cwli_16 tutorial_ensadv tutorial/train_ensadv 6705

# predict_and_compare thin_fgsm_8 thin log
# predict_and_compare thin_fgsm_12 thin log
# predict_and_compare thin_fgsm_16 thin log
# predict_and_compare wide_fgsm_8 wide log_wide
# predict_and_compare wide_fgsm_12 wide log_wide
# predict_and_compare wide_fgsm_16 wide log_wide
# predict_and_compare tutorial_fgsm_8 tutorial tutorial/train
# predict_and_compare tutorial_fgsm_12 tutorial tutorial/train
# predict_and_compare tutorial_fgsm_16 tutorial tutorial/train

# predict_and_compare thin_cwli_8 thin log 6705
# predict_and_compare thin_cwli_12 thin log 6705
# predict_and_compare thin_cwli_16 thin log 6705
# predict_and_compare wide_cwli_8 wide log_wide 6705
# predict_and_compare wide_cwli_12 wide log_wide 6705
# predict_and_compare wide_cwli_16 wide log_wide 6705
# predict_and_compare tutorial_cwli_8 tutorial tutorial/train 6705
# predict_and_compare tutorial_cwli_12 tutorial tutorial/train 6705
# predict_and_compare tutorial_cwli_16 tutorial tutorial/train 6705

# predict_and_compare thin_opt_8 thin log 6705
# predict_and_compare thin_opt_8 tutorial_adv tutorial/train_adv 6705
# predict_and_compare thin_opt_8 tutorial_ensadv tutorial/train_ensadv 6705
# predict_and_compare thin_opt_12 thin log 6705
# predict_and_compare thin_opt_12 tutorial_adv tutorial/train_adv 6705
# predict_and_compare thin_opt_12 tutorial_ensadv tutorial/train_ensadv 6705
# predict_and_compare thin_opt_16 thin log 6705
# predict_and_compare thin_opt_16 tutorial_adv tutorial/train_adv 6705
# predict_and_compare thin_opt_16 tutorial_ensadv tutorial/train_ensadv 6705

# predict_and_compare wide_opt_8 wide log_wide 6705
# predict_and_compare wide_opt_8 thin_adv log_adv 6705
# predict_and_compare wide_opt_8 thin_ensadv log_ensadv 6705
# predict_and_compare wide_opt_12 wide log_wide 6705
# predict_and_compare wide_opt_12 thin_adv log_adv 6705
# predict_and_compare wide_opt_12 thin_ensadv log_ensadv 6705
# predict_and_compare wide_opt_16 wide log_wide 6705
# predict_and_compare wide_opt_16 thin_adv log_adv 6705
# predict_and_compare wide_opt_16 thin_ensadv log_ensadv 6705

# predict_and_compare tutorial_opt_8 tutorial tutorial/train 6705
# predict_and_compare tutorial_opt_8 wide_adv log_adv_wide 6705
# predict_and_compare tutorial_opt_8 wide_ensadv log_ensadv_wide 6705
# predict_and_compare tutorial_opt_12 tutorial tutorial/train 6705
# predict_and_compare tutorial_opt_12 wide_adv log_adv_wide 6705
# predict_and_compare tutorial_opt_12 wide_ensadv log_ensadv_wide 6705
# predict_and_compare tutorial_opt_16 tutorial tutorial/train 6705
# predict_and_compare tutorial_opt_16 wide_adv log_adv_wide 6705
# predict_and_compare tutorial_opt_16 wide_ensadv log_ensadv_wide 6705
