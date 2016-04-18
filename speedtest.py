import numpy as np
import time
import theano
import theano.tensor as T

import lasagne
from lasagne.layers import InputLayer, MergeLayer, Layer

floatX = theano.config.floatX = 'float32'

from raccoon.layers.reccurrent import GRULayer




mode = False

bs = 32
n_hidden = 256
emb_size = 200
seq_l = 1000

seq_con1 = T.alloc(0.0, bs, seq_l, emb_size).astype(dtype='float32')
mask1 = T.ones((bs, seq_l), dtype='float32')

seq_con = T.alloc(0.0, seq_l, bs, emb_size).astype(dtype='float32')
mask = T.ones((seq_l, bs), dtype='float32')

if mode:
    x = lasagne.layers.InputLayer((None, None, emb_size), input_var=seq_con1)
    mask = lasagne.layers.InputLayer((None, None), input_var=mask1)

    y = lasagne.layers.GRULayer(x, n_hidden, mask_input=mask)

    out = lasagne.layers.get_output(y)

else:
    l = GRULayer(emb_size, n_hidden, lasagne.init.GlorotUniform())
    out, _ = l.apply(seq_con, mask,
                  T.alloc(0.0, bs, n_hidden).astype(dtype='float32'))

print 'compiling...',
f = theano.function([], out)
print 'done'


b = time.clock()
for i in range(3):
    f()
print time.clock() - b