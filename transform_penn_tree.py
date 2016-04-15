import os
from itertools import count
import cPickle

import numpy


path = '/Users/adeb/data/PennTreebankCorpus'

data = numpy.load(os.path.join(path, 'penntree_char_and_word.npz'))
train = data['train_words']
valid = data['valid_words']
test = data['test_words']

ordered_tokens = numpy.argsort(numpy.bincount(train))
index_mapping = {old: new for old, new in zip(ordered_tokens, count())}
new_unk_idx = index_mapping.get(591)
train = [index_mapping.get(old, new_unk_idx) for old in train]
valid = [index_mapping.get(old, new_unk_idx) for old in valid]
test = [index_mapping.get(old, new_unk_idx) for old in test]
counts = numpy.bincount(train).astype('float32')
counts /= counts.sum()

vocab = numpy.load(open(os.path.join(path, "dictionaries.npz")))['unique_words']

vocab = {vocab[i]:j for i, j in index_mapping.iteritems()}

cPickle.dump(vocab, open('pt_vocab.pkl', 'w'))
cPickle.dump((train, valid, test, vocab), open('pt_data.pkl', 'w'))