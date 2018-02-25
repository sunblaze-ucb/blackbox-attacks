# Query-based black-box attacks

This repository contains code to reproduce results from the paper:

**Exploring the Space of Black-box Attacks on Deep Neural Networks** <br>
*Arjun Nitin Bhagoji, Warren He, Bo Li and Dawn Song* <br>
ArXiv report: https://arxiv.org/abs/1712.09491

#### REQUIREMENTS

The code was tested with Python 2.7.12, Tensorflow 1.3.0 and Keras 1.2.2.

## EXPERIMENTS

### TRAINING *MNIST* MODELS

#### STANDARD TRAINING
To train Models A through D on the MNIST dataset, create a ```models``` directory and run ```train.py``` as follows:
```
python train.py models/modelA --type=0 --num_epochs=6 
python train.py models/modelB --type=1 --num_epochs=6 
python train.py models/modelC --type=2 --num_epochs=6 
python train.py models/modelD --type=3 --num_epochs=6 
```
#### ADVERSARIAL TRAINING
To obtain variants of Models A through D which are trained using standard adversarial training, run ```train_adv.py``` as follows:
```
python train_adv.py models/modelA_adv --type=0 --num_epochs=12 
python train_adv.py models/modelB_adv --type=1 --num_epochs=12
python train_adv.py models/modelC_adv --type=2 --num_epochs=12
python train_adv.py models/modelD_adv --type=3 --num_epochs=12
```
Note that the number of epochs is increased from 6 to 12. The magnitude of the adversarial perturbation used for the training images can be controlled with the `-eps` flag. The default value used for MNIST is 0.3.

#### ENSEMBLE ADVERSARIAL TRAINING
The ```train_adv.py``` script can also be run to obtain variants of Models A through D trained using [ensemble adversarial training](https://arxiv.org/abs/1705.07204). For example, to train a variant of Model A that uses adversarial samples generated from Models A, B and C, as well samples generated from the current state of the model (as in standard adversarial training), run:
```
python train_adv.py models/modelA_ens models/modelA models/modelC models/modelD --type=0 --num_epochs=12
```

#### ITERATIVE ADVERSARIAL TRAINING
Using the `--iter` flag with `train_adv.py` allows for the training of variants of Models A through D using [iterative adversarial training](https://arxiv.org/abs/1706.06083). For example, a variant of Model A with iterative adversarial training can be trained as follows:
```
python train_adv.py models/modelA_adv --type=0 --iter=1 --num_epochs=64
```
Note that this form of training needs a much higher number of epochs for the training to converge. Iterative adversarial samples are generated using 40 steps of magnitude 0.01 each by default. This can be changed in the `train_adv.py` script. The maximum perturbation magnitude is still set to 0.3. To train using only the adversarial loss, set the `--ben` flag to 0. 

### PRETRAINED *CIFAR-10* MODELS
CIFAR-10 models are trained using the same techniques.
We have uploaded a set of [pretrained weights](https://berkeley.box.com/s/obtatlkt9tppvemb3ufxpxt3a35n0p0l).

### ATTACKING _MNIST_ MODELS
For all attacks except the _Random Pertubations_ attack, setting `--targeted_flag=1` enables a targeted attack to be carried out. By default, the target for each sample is chosen uniformly at random from the set of class labels except the true label of that sample. For attacks that generate adversarial examples (not the baseline attacks), two loss functions can be used: the standard cross-entropy loss (use `--loss_type=xent`) and a [logit-based loss](https://arxiv.org/abs/1608.04644) (use `--loss_type=cw`). 

#### BASELINE ATTACKS
In order to carry out an _untargeted Difference of Means_ attack (on Model A for example), run the `baseline_attacks.py` script as follows:
```
python baseline_attacks.py models/modelA
``` 
This will run the attack for a pre-specified list of perturbation values. For attacks constrained using the *infinity-norm*, the maximum perturbation value is 0.5 and for attacks constrained using the *2-norm*, it is 9.0. To carry out an _untargeted Random Perturbations_ attack, the `--alpha` parameter is set to 0.6 (infinity-norm constraint) or 9.1 (2-norm constraint).

#### TRANSFERABILITY-BASED ATTACKS
To carry out a transferability-based attack on a single model (say Model B) using FGS adversarial examples generated for another single model (say Model A), run the `transfer_attack_w_ensemble.py` script as follows:
```
python transfer_attack_w_ensemble.py fgs models/modelA --target_model=models/modelB
```
An [ensemble-based transferability](https://arxiv.org/abs/1611.02770) attack can also carried out. For example, if the ensemble of local models consists of Models A, C and D, and the model being attacked is Model B, run `transfer_attack_w_ensemble.py` as follows:
```
python transfer_attack_w_ensemble.py fgs models/modelA models/modelC models/modelC --target_model=models/modelB
``` 
The `transfer_attack_w_ensemble.py` script also supports a number of other attacks including Iterative FGS, FGS with an initial random step and the [Carlini-Wagner](https://arxiv.org/abs/1608.04644) attack, all of which can be carried out for either a single local model or an ensemble of models.

If the `--target_model` option is not specified, then just the white-box attack success rates will be reported. Note that if the perturbation magnitude is not specified using the `--eps` option, then a default set of perturbation values will be used.

#### QUERY-BASED ATTACKS
These attacks are carried out directly on a target model assuming query access to the output probabilities. An _untargeted query-based attack_ with no query reduction using _Single Step Finite Differences_ can be carried out on Model A as follows:
```
python query_based_attack.py models/modelA --method=query_based_un
```
A parameter `--delta` can be used to control the perturbations in input space used to estimate gradients. It is set to a default value of 0.01. To run a targeted version, set the `--method` option to 'query_based'. The_untargeted Iterative Finite Differences_ attack can be run as follows:
```
python query_based_attack.py models/modelA --method=query_based_un_iter
```
The number of iterations is set using the `--num_iter` flag and the step size per iteration is set using the `--beta` flag. The default values of these are 40 and 0.1 respectively. Targeted attacks can be run by setting the `--method` option to 'query_based_iter'. 

To run an attack using the technique of _Simultaneous Perturbation Stochastic Approximation (SPSA)_, the `--method` option is set to 'spsa_iter' for targeted attacks and to 'spsa_un_iter' for untargeted attacks. 

##### USING QUERY-REDUCTION TECHNIQUES
To reduce the number of queries, two methods are implemented in the `query_based_attack.py` script. These can be used along with any of the Finite Difference methods ('query_based', 'query_based_un', 'query_based_iter' and 'query_based_un_iter') To use the _Random Grouping_ method with 8 pixels grouped together, for example, with the untargeted Single Step Gradient Estimation method, run
```
python query_based_attack.py models/modelA --method=query_based_un --group_size=8
```
Similarly, to use the _PCA component_ based query reduction with 100 components, for example, with the same attack as above, run
```
python query_based_attack.py models/modelA --method=query_based_un --num_comp=100
```
These query-reduction techniques can be used with targeted, untargeted, Single-Step and Iterative Gradient Estimation methods.

### ATTACKS ON CLARIFAI
To run attacks on models hosted by Clarifai, first follow the instructions given [here](https://clarifai-python.readthedocs.io/en/latest/install/) to install their Python client. You will need to obtain _your own API key_ and set it using `clarifai config`. The two models currently supported for attack are the 'moderation' and 'nsfw-v1.0' models. To obtain an adversarial example for the 'moderation' model starting with _my_image.jpg_, run the [`attack_clarifai.py`](clarifai/attack_clarifai.py) script as follows:
```
python attack_clarifai.py my_image --target_model=moderation
``` 
The default attack used is _Gradient Estimation with query reduction using Random Grouping_. The available options are the magnitude of perturbation (`--eps`), number of iterations (`--num_iter`), group size for random grouping (`--group_size`) and gradient estimation parameter (`--delta`). Only the logit loss is used since it is found to perform well. The target image must be an RGB image in the JPEG format. The [`Resizing_clarifai.ipynb`](clarifai/Resizing_clarifai.ipynb) notebook allows for interactive re-sizing of images in case the target image is too large.


#### CONTACT
This repository is under active development. Questions and suggestions can be sent to abhagoji@princeton.edu
