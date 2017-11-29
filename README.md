# Query-based black-box attacks

This repository contains code to reproduce results from the paper:

**Exploring the Space of Black-box Attacks on DeepNeural Networks** <br>
*Arjun Nitin Bhagoji, Warren He, Bo Li and Dawn Song* <br>
ArXiv report:

<br>

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

#### ENSEMBLE ADVERSARIAL TRAINING

#### ITERATIVE ADVERSARIAL TRAINING

### PRETRAINED *CIFAR-10* MODELS
CIFAR-10 models are trained using the same techniques.
We have uploaded a set of [pretrained weights](https://berkeley.box.com/s/obtatlkt9tppvemb3ufxpxt3a35n0p0l).

### BASELINE ATTACKS

### TRANSFERABILITY-BASED ATTACKS

### QUERY-BASED ATTACKS

#### USING QUERY-REDUCTION TECHNIQUES

#### ATTACKS ON CLARIFAI



#### CONTACT
Questions and suggestions can be sent to abhagoji@princeton.edu
