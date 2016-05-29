# Fast attentive reader from "Teaching Machines to Read and Comprehend"
Lasagne/Theano implementation of the attentive reader of the following paper from Google DeepMind in Theano/Lasagne.

[Teaching Machines to Read and Comprehend](http://arxiv.org/abs/1506.03340),  
Karl Moritz Hermann, Tomáš Kočiský, Edward Grefenstette, Lasse Espeholt, Will Kay, Mustafa Suleyman, Phil Blunsom,  
NIPS 2015

Our attentive reader architecture/hyperparameters are different from Deepmind's, it is considerably faster to train, **reaching 62.1% accuracy in only 4 hours** (green curve below).

![img](https://raw.githubusercontent.com/adbrebs/rnn_reader/master/training_profile.png "Guillaume Apollinaire")

# Instructions

0) Clone the repository 

1) Follow the instructions steps of https://github.com/thomasmesnard/DeepMind-Teaching-Machines-to-Read-and-Comprehend to download and process the data. 

2) Create a $DATA_PATH env variable with the path of the dataset folder. More precisely the dataset folder should have the following structure: 
$DATA_PATH > deepmind-qa > cnn > questions and stats folder 

3) Go to the cloned repository and run ``python main.py -s config/big/attention_softmax.py``.

It should take **about 4 hours** to reach a validation performance of 62% on a Titan X.

# Differences between deepmind's model and ours
If the attentive reader mechanism is the same, there are many architecture differences with Deepmind's model. 
In particular:

- we use GRUs instead of LSTMS
- we don't use dropout
- we use ADAM as gradient descent rule
- our model
- no bidirectional layer, but a single and small (256 units) GRU layer

# Requirements

- Theano
- Lasagne
- Fuel (for the data pipeline)
- [Raccoon](https://github.com/adbrebs/raccoon) (to monitor training)

# Credit

Our code relies on the fuel data extensions developed by Thomas Mesnard, Alex Auvolat and Étienne Simon: https://github.com/thomasmesnard/DeepMind-Teaching-Machines-to-Read-and-Comprehend

