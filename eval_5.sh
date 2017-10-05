#!/bin/sh -ex

### targeted attacks

export CUDA_VISIBLE_DEVICES=4
export TF_CPP_MIN_LOG_LEVEL=2

predict() { # src dst ckpt_dir
	python predict_file.py preds_$1_$2.npy test_$1.npy $3
}


predict_and_compare() { # src dst ckpt_dir [offset]
	predict $1 $2 $3
	python compare_preds_targeted.py preds_$1_$2.npy $4
}

# 8 fgsm, xent loss

# predict_and_compare thin_fgsm_targeted_8 thin log
# predict_and_compare thin_fgsm_targeted_8 thin_adv log_adv
# predict_and_compare thin_fgsm_targeted_8 thin_ensadv log_ensadv
# predict_and_compare thin_fgsm_targeted_8 wide log_wide
# predict_and_compare thin_fgsm_targeted_8 wide_adv log_adv_wide
# predict_and_compare thin_fgsm_targeted_8 wide_ensadv log_ensadv_wide
# predict_and_compare thin_fgsm_targeted_8 tutorial tutorial/train
# predict_and_compare thin_fgsm_targeted_8 tutorial_adv tutorial/train_adv
# predict_and_compare thin_fgsm_targeted_8 tutorial_ensadv tutorial/train_ensadv

# predict_and_compare wide_fgsm_targeted_8 thin log
# predict_and_compare wide_fgsm_targeted_8 thin_adv log_adv
# predict_and_compare wide_fgsm_targeted_8 thin_ensadv log_ensadv
predict_and_compare wide_fgsm_targeted_8 wide log_wide
predict_and_compare wide_fgsm_targeted_8 tutorial tutorial/train

# 8 fgsm, logit loss

# predict_and_compare thin_fgsmlogit_targeted_8 thin log
# predict_and_compare thin_fgsmlogit_targeted_8 thin_adv log_adv
# predict_and_compare thin_fgsmlogit_targeted_8 thin_ensadv log_ensadv
# predict_and_compare thin_fgsmlogit_targeted_8 wide log_wide
# predict_and_compare thin_fgsmlogit_targeted_8 wide_adv log_adv_wide
# predict_and_compare thin_fgsmlogit_targeted_8 wide_ensadv log_ensadv_wide
# predict_and_compare thin_fgsmlogit_targeted_8 tutorial tutorial/train
# predict_and_compare thin_fgsmlogit_targeted_8 tutorial_adv tutorial/train_adv
# predict_and_compare thin_fgsmlogit_targeted_8 tutorial_ensadv tutorial/train_ensadv

predict_and_compare wide_fgsmlogit_targeted_8 thin log
predict_and_compare wide_fgsmlogit_targeted_8 thin_adv log_adv
predict_and_compare wide_fgsmlogit_targeted_8 thin_ensadv log_ensadv
predict_and_compare wide_fgsmlogit_targeted_8 wide log_wide
predict_and_compare wide_fgsmlogit_targeted_8 tutorial tutorial/train

# 8 opt

# predict_and_compare thin_opt_targeted_8 thin log 6705
# predict_and_compare thin_opt_targeted_8 thin_adv log_adv 6705
# predict_and_compare thin_opt_targeted_8 thin_ensadv log_ensadv 6705
# predict_and_compare thin_opt_targeted_8 wide log_wide 6705
# predict_and_compare thin_opt_targeted_8 wide_adv log_adv_wide 6705
# predict_and_compare thin_opt_targeted_8 wide_ensadv log_ensadv_wide 6705
# predict_and_compare thin_opt_targeted_8 tutorial tutorial/train 6705
# predict_and_compare thin_opt_targeted_8 tutorial_adv tutorial/train_adv 6705
# predict_and_compare thin_opt_targeted_8 tutorial_ensadv tutorial/train_ensadv 6705

# 8 iterative gradient sign

# predict_and_compare thin_itergs_targeted_8 thin log
# predict_and_compare thin_itergs_targeted_8 thin_adv log_adv
# predict_and_compare thin_itergs_targeted_8 thin_ensadv log_ensadv
# predict_and_compare thin_itergs_targeted_8 wide log_wide
# predict_and_compare thin_itergs_targeted_8 wide_adv log_adv_wide
# predict_and_compare thin_itergs_targeted_8 wide_ensadv log_ensadv_wide
# predict_and_compare thin_itergs_targeted_8 tutorial tutorial/train
# predict_and_compare thin_itergs_targeted_8 tutorial_adv tutorial/train_adv
# predict_and_compare thin_itergs_targeted_8 tutorial_ensadv tutorial/train_ensadv

predict_and_compare wide_itergs_targeted_8 thin log
predict_and_compare wide_itergs_targeted_8 thin_adv log_adv
predict_and_compare wide_itergs_targeted_8 thin_ensadv log_ensadv
predict_and_compare wide_itergs_targeted_8 wide log_wide
predict_and_compare wide_itergs_targeted_8 tutorial tutorial/train

# 8 iterative gradient sign, logit loss

# predict_and_compare thin_itergslogit_targeted_8 thin log
# predict_and_compare thin_itergslogit_targeted_8 thin_adv log_adv
# predict_and_compare thin_itergslogit_targeted_8 thin_ensadv log_ensadv
# predict_and_compare thin_itergslogit_targeted_8 wide log_wide
# predict_and_compare thin_itergslogit_targeted_8 wide_adv log_adv_wide
# predict_and_compare thin_itergslogit_targeted_8 wide_ensadv log_ensadv_wide
# predict_and_compare thin_itergslogit_targeted_8 tutorial tutorial/train
# predict_and_compare thin_itergslogit_targeted_8 tutorial_adv tutorial/train_adv
# predict_and_compare thin_itergslogit_targeted_8 tutorial_ensadv tutorial/train_ensadv

predict_and_compare wide_itergslogit_targeted_8 thin log
predict_and_compare wide_itergslogit_targeted_8 thin_adv log_adv
predict_and_compare wide_itergslogit_targeted_8 thin_ensadv log_ensadv
predict_and_compare wide_itergslogit_targeted_8 wide log_wide
predict_and_compare wide_itergslogit_targeted_8 tutorial tutorial/train

# 16

# predict_and_compare thin_fgsm_targeted_16 thin log
# predict_and_compare thin_fgsm_targeted_16 thin_adv log_adv
# predict_and_compare thin_fgsm_targeted_16 thin_ensadv log_ensadv
# predict_and_compare thin_fgsm_targeted_16 wide log_wide
# predict_and_compare thin_fgsm_targeted_16 wide_adv log_adv_wide
# predict_and_compare thin_fgsm_targeted_16 wide_ensadv log_ensadv_wide
# predict_and_compare thin_fgsm_targeted_16 tutorial tutorial/train
# predict_and_compare thin_fgsm_targeted_16 tutorial_adv tutorial/train_adv
# predict_and_compare thin_fgsm_targeted_16 tutorial_ensadv tutorial/train_ensadv

# predict_and_compare thin_fgsmlogit_targeted_16 thin log
# predict_and_compare thin_fgsmlogit_targeted_16 thin_adv log_adv
# predict_and_compare thin_fgsmlogit_targeted_16 thin_ensadv log_ensadv
# predict_and_compare thin_fgsmlogit_targeted_16 wide log_wide
# predict_and_compare thin_fgsmlogit_targeted_16 wide_adv log_adv_wide
# predict_and_compare thin_fgsmlogit_targeted_16 wide_ensadv log_ensadv_wide
# predict_and_compare thin_fgsmlogit_targeted_16 tutorial tutorial/train
# predict_and_compare thin_fgsmlogit_targeted_16 tutorial_adv tutorial/train_adv
# predict_and_compare thin_fgsmlogit_targeted_16 tutorial_ensadv tutorial/train_ensadv
