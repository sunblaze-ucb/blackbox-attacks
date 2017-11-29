# Query-based black-box attacks

This repository contains code to reproduce results from the paper:

**Exploring the Space of Black-box Attacks on Deep Neural Networks** <br>
*Arjun Nitin Bhagoji, Warren He, Bo Li and Dawn Song* <br>
ArXiv report:

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

### ATTACKING _MNIST_ MODELS

#### BASELINE ATTACKS
In order to carry out an _untargeted Difference of Means_ attack (on Model A for example), run the `baseline_attacks.py` script as follows:
```
python baseline_attacks.py models/modelA
``` 
This will run the attack for a pre-specified list of perturbation values. For attacks constrained using the *infinity-norm*, the maximum perturbation value is 0.5 and for attacks constrained using the *2-norm*, it is 9.0. To carry out a _targeted Difference of Means_ attack, set `--targeted-flag=1`.



#### TRANSFERABILITY-BASED ATTACKS

#### QUERY-BASED ATTACKS

##### USING QUERY-REDUCTION TECHNIQUES

### ATTACKS ON CLARIFAI



#### CONTACT
This repository is under active development. Questions and suggestions can be sent to abhagoji@princeton.edu
