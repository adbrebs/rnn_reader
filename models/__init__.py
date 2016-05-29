import numpy as np

import theano
import theano.tensor as T

from lasagne.layers import InputLayer, get_output
from lasagne.objectives import categorical_accuracy, categorical_crossentropy

floatX = theano.config.floatX = 'float32'


class ReaderModel:
    def __init__(self, vocab_size, n_entities, embedding_size, residual=False,
                 depth_rnn=1, grad_clipping=10, skip_connections=False,
                 bidir=False, dropout=False):
        self.vocab_size = vocab_size
        self.n_entities = n_entities
        self.embedding_size = embedding_size
        self.residual = residual
        self.depth_rnn = depth_rnn
        self.grad_clipping = grad_clipping
        self.skip_connections = skip_connections
        self.bidir = bidir
        self.dropout = dropout
        self.net = None

        self.tg = T.vector('tg', 'int32')
        self.cand = T.matrix('candidates', 'int32')
        self.cand_mask = T.matrix('candidates_mask', floatX)

        self.vars = [
            self.tg,
            self.cand,
            self.cand_mask
        ]

        self.in_cand = InputLayer((None, None), self.cand)
        self.in_cand_mask = InputLayer((None, None), self.cand_mask)

        self.init_virtual()

        self.create_debug_values()

    def init_virtual(self):
        raise NotImplemented

    def compute_cost(self, deterministic=False):
        output = get_output(self.net, deterministic=deterministic)

        cost = categorical_crossentropy(output, self.tg).mean()
        cost.name = 'negll'

        accuracy = categorical_accuracy(output, self.tg).mean()
        accuracy.name = 'accuracy'

        return cost, accuracy

    def create_debug_values(self):
        """
        Used for debug with tag.test_value
        """
        batch_size = 16
        n_entities = self.n_entities
        n_candidates = 5

        self.tg.tag.test_value = np.random.randint(
            0, n_entities, (batch_size,)).astype('int32')

        self.cand.tag.test_value = np.repeat(
            np.arange(n_candidates, dtype='int32')[None, :],
            batch_size, axis=0)
        self.cand_mask.tag.test_value = np.ones((batch_size, n_candidates),
                                                dtype=floatX)

        self.create_debug_values_virtual()

    def create_debug_values_virtual(self):
        raise NotImplemented


class ReaderTwoSeqModel(ReaderModel):
    """
    Reader which separates context from the question in two sequences
    """
    def init_virtual(self):

        self.seq_con = T.matrix('seq_cont', 'int32')
        self.seq_con_mask = T.matrix('seq_cont_mask', floatX)
        self.seq_que = T.matrix('seq_quest', 'int32')
        self.seq_que_mask = T.matrix('seq_quest_mask', floatX)

        self.vars = [
            self.seq_con,
            self.seq_con_mask,
            self.seq_que,
            self.seq_que_mask] + self.vars

        self.in_con = InputLayer((None, None), self.seq_con)
        self.in_con_mask = InputLayer((None, None), self.seq_con_mask)
        self.in_que = InputLayer((None, None), self.seq_que)
        self.in_que_mask = InputLayer((None, None), self.seq_que_mask)

    def create_debug_values_virtual(self):

        seq_con_length = 3
        seq_que_length = 2
        batch_size = 16
        vocab_size = self.vocab_size

        self.seq_con.tag.test_value = np.random.randint(0, vocab_size, (
            batch_size, seq_con_length)).astype('int32')
        self.seq_con_mask.tag.test_value = np.ones(
            (batch_size, seq_con_length), dtype=floatX)

        self.seq_que.tag.test_value = np.random.randint(0, vocab_size, (
            batch_size, seq_que_length)).astype('int32')
        self.seq_que_mask.tag.test_value = np.ones(
            (batch_size, seq_que_length), dtype=floatX)


class ReaderOneSeqModel(ReaderModel):
    """
    Reader which concates the context the question sequences
    """
    def init_virtual(self):

        self.seq = T.matrix('seq_cont_que', 'int32')
        self.seq_mask = T.matrix('seq_cont_que_mask', floatX)

        self.vars = [self.seq, self.seq_mask] + self.vars

        self.in_seq = InputLayer((None, None), self.seq)
        self.in_mask = InputLayer((None, None), self.seq_mask)

    def create_debug_values_virtual(self):

        seq_length = 3
        batch_size = 16
        vocab_size = self.vocab_size

        self.seq.tag.test_value = np.random.randint(0, vocab_size, (
            batch_size, seq_length)).astype('int32')
        self.seq_mask.tag.test_value = np.ones(
            (batch_size, seq_length), dtype=floatX)


