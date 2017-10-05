#!/bin/sh -ex

### untargeted attacks

export CUDA_VISIBLE_DEVICES=4
export TF_CPP_MIN_LOG_LEVEL=2

predict() { # src dst ckpt_dir
	python predict_file.py preds_$1_$2.npy test_$1.npy $3
}


predict_and_compare() { # src dst ckpt_dir [offset]
	predict $1 $2 $3
	python compare_preds.py preds_orig_$2.npy preds_$1_$2.npy $4
}

# 8 fgsm, xent loss

# predict_and_compare thin_fgsm_8 wide log_wide

# 8 opt

# predict_and_compare thin_opt_8 wide log_wide 6705

# 8 fgsm, logit loss

# predict_and_compare thin_fgsmlogit_8 thin log
# predict_and_compare thin_fgsmlogit_8 wide log_wide
# predict_and_compare thin_fgsmlogit_8 tutorial tutorial/train

# 8 iterative gradient sign

# predict_and_compare thin_itergs_8 thin log
# predict_and_compare thin_itergs_8 wide log_wide
# predict_and_compare thin_itergs_8 tutorial tutorial/train

# 8 iterative gradient sign, logit loss

# predict_and_compare thin_itergslogit_8 thin log
# predict_and_compare thin_itergslogit_8 wide log_wide
# predict_and_compare thin_itergslogit_8 tutorial tutorial/train

compare2() { # src dst [offset]
	python compare_preds_2.py preds_$1_$2.npy $3
}

# compare2 thin_fgsm_8 thin
# compare2 thin_fgsm_8 wide
# compare2 thin_fgsm_8 tutorial

# compare2 thin_fgsmlogit_8 thin
# compare2 thin_fgsmlogit_8 wide
# compare2 thin_fgsmlogit_8 tutorial

# compare2 thin_itergs_8 thin
# compare2 thin_itergs_8 wide
# compare2 thin_itergs_8 tutorial

# compare2 thin_itergslogit_8 thin
# compare2 thin_itergslogit_8 wide
# compare2 thin_itergslogit_8 tutorial

predict_and_compare2() { # src dst ckpt_dir [offset]
	predict $1 $2 $3
	python compare_preds_2.py preds_$1_$2.npy $4
}

predict_and_compare2 wide_fgsm_8 thin log
predict_and_compare2 wide_fgsm_8 thin_adv log_adv
predict_and_compare2 wide_fgsm_8 thin_ensadv log_ensadv
predict_and_compare2 wide_fgsm_8 wide log_wide
predict_and_compare2 wide_fgsm_8 tutorial tutorial/train

predict_and_compare2 wide_fgsmlogit_8 thin log
predict_and_compare2 wide_fgsmlogit_8 thin_adv log_adv
predict_and_compare2 wide_fgsmlogit_8 thin_ensadv log_ensadv
predict_and_compare2 wide_fgsmlogit_8 wide log_wide
predict_and_compare2 wide_fgsmlogit_8 tutorial tutorial/train

predict_and_compare2 wide_itergs_8 thin log
predict_and_compare2 wide_itergs_8 thin_adv log_adv
predict_and_compare2 wide_itergs_8 thin_ensadv log_ensadv
predict_and_compare2 wide_itergs_8 wide log_wide
predict_and_compare2 wide_itergs_8 tutorial tutorial/train

predict_and_compare2 wide_itergslogit_8 thin log
predict_and_compare2 wide_itergslogit_8 thin_adv log_adv
predict_and_compare2 wide_itergslogit_8 thin_ensadv log_ensadv
predict_and_compare2 wide_itergslogit_8 wide log_wide
predict_and_compare2 wide_itergslogit_8 tutorial tutorial/train
