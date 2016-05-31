# Fast attentive reader from "Teaching Machines to Read and Comprehend"
Lasagne/Theano implementation of the attentive reader of the following paper from Google DeepMind in Theano/Lasagne.

[Teaching Machines to Read and Comprehend](http://arxiv.org/abs/1506.03340),  
Karl Moritz Hermann, Tomáš Kočiský, Edward Grefenstette, Lasse Espeholt, Will Kay, Mustafa Suleyman, Phil Blunsom,  
NIPS 2015

Our attentive reader architecture/hyperparameters are different from Deepmind's, it is considerably faster to train, **reaching 62.1% accuracy in only 4-5 hours** (green curve below). Learning starts straight away, the plateau phase is very short.

![img](https://raw.githubusercontent.com/adbrebs/rnn_reader/master/training_profile.png "Raccoon demon")

# Instructions

0) Clone the repository 

1) Follow the instructions steps of https://github.com/thomasmesnard/DeepMind-Teaching-Machines-to-Read-and-Comprehend to download and process the data. 

2) Create a $DATA_PATH env variable with the path of the dataset folder. More precisely the dataset folder should have the following structure: 
$DATA_PATH > deepmind-qa > cnn > questions and stats folder 

3) Go to the cloned repository and run ``python main.py -s config/big/attention_softmax.py``.

It should take **about 4-5 hours** to reach a validation performance of 62% on a Titan X.

# Differences between deepmind's model and ours
If the attentive reader mechanism is the same, there are several architecture differences compared to Deepmind's model. 
In particular:

- we use GRUs instead of LSTMS,
- we don't use dropout,
- we use ADAM as gradient descent rule,
- a single small layer of GRUs (256 units),
- no bidirectional layer,
- no skip connections,
- no weight noise,
- vocabulary is limited to 30K words.

The config file of our architecture:
https://github.com/adbrebs/rnn_reader/blob/master/config/big/attention_softmax.py

# Requirements

- Theano
- Lasagne
- Fuel (for the data pipeline)
- [Raccoon](https://github.com/adbrebs/raccoon) (to monitor training)

# Credit

Our code relies on the fuel data extensions developed by Thomas Mesnard, Alex Auvolat and Étienne Simon: https://github.com/thomasmesnard/DeepMind-Teaching-Machines-to-Read-and-Comprehend

