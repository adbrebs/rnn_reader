import numpy as np

import theano
import theano.tensor as T

from lasagne.layers import (
    InputLayer, MergeLayer, Layer, ElemwiseMergeLayer, dimshuffle,
    ReshapeLayer, DenseLayer, ElemwiseSumLayer, concat, DropoutLayer,
    get_output)
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


class SequenceSoftmax(MergeLayer):
    """
    Computes a softmax over a sequence associated with a mask.

    Parameters
    ----------
    layer: layer of shape (batch_size, sequence_length, ...)
    layer_mask: layer of shape (batch_size, sequence_length, ...)

    Notes
    -----
    This layer has the same output shape as the parameter layer
    layer and layer_mask should have compatible shapes.
    """
    def __init__(self, layer, layer_mask, seq_axis=1, name=None):
        MergeLayer.__init__(self, [layer, layer_mask], name=name)
        self.seq_axis = seq_axis

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, **kwargs):
        att, mask = inputs

        att = T.exp(att - T.max(att, axis=self.seq_axis, keepdims=True))
        att = att * mask
        att /= T.sum(att, axis=self.seq_axis, keepdims=True)

        return att


class EfficientAttentionLayer(MergeLayer):
    """
    Layer of shape (batch_size, n_features)

    Parameters
    ----------
    attended_layer: layer of shape (batch_size, seq_length, n_features)
    attended_layer_mask: layer of shape (batch_size, seq_length, n_features)
    condition_layer: layer of shape (batch_size, n_features)
    """
    def __init__(self, attended_layer, attended_layer_mask,
                 condition_layer, name=None):
        MergeLayer.__init__(self, [attended_layer, attended_layer_mask,
                                   condition_layer], name=name)

    def get_output_shape_for(self, input_shapes):
        attended_layer_shape = input_shapes[0]
        return attended_layer_shape[0], attended_layer_shape[-1]

    def get_output_for(self, inputs, **kwargs):

        # seq_input: (batch_size, seq_size, n_hidden_con)
        # seq_mask: (batch_size, seq_size)
        # condition: (batch_size, n_hidden_con)
        seq_input, seq_mask, condition = inputs

        seq_input *= T.shape_padright(seq_mask)
        # (batch_size, n_hidden_question, n_hidden_question)
        covariance = T.batched_dot(seq_input.dimshuffle(0, 2, 1), seq_input)
        # (batch_size, n_hidden_question), equivalent to the following line:
        # att = T.sum(covariance * condition.dimshuffle((0, 'x', 1)), axis=2)
        att = T.batched_dot(covariance, condition.dimshuffle((0, 1)))

        att = 1000 * att / T.sum(seq_mask, axis=1, keepdims=True)
        # norm2_att = T.sum(att * condition, axis=1, keepdims=True)
        # att = 1000 * att / norm2_att

        return att


class CandidateOutputLayer(MergeLayer):
    """
    Layer of shape (batch_size, n_outputs)
    Parameters
    ----------
    output_layer: layer of shape (batch_size, n_outputs)
    candidate_layer: layer of shape (batch_size, max_n_candidates)
    candidate_mask_layer: layer of shape (batch_size, max_n_candidates)
    """
    def __init__(self, output_layer, candidate_layer, candidate_mask_layer,
                 non_linearity=T.nnet.softmax, name=None):
        MergeLayer.__init__(self, [output_layer, candidate_layer,
                                   candidate_mask_layer], name=name)
        self.non_linearity = non_linearity

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, **kwargs):

        # out: (batch_size, n_entities)
        # cand: (batch_size, n_candidates)
        # cand_mask: (batch_size, n_candidates)
        out, cand, cand_mask = inputs

        n_entities = self.input_shapes[0][1]
        is_candidate = T.eq(
            T.arange(n_entities, dtype='int32')[None, None, :],
            T.switch(cand_mask, cand,
                     -T.ones_like(cand))[:, :, None]).sum(axis=1)

        out = T.switch(is_candidate, out, -1000 * T.ones_like(out))

        return self.non_linearity(out)


class ForgetSizeLayer(Layer):
    def __init__(self, incoming, axis=-1, **kwargs):
        Layer.__init__(self, incoming, **kwargs)
        self.axis = axis

    def get_output_for(self, input, **kwargs):
        return input

    def get_output_shape_for(self, input_shape):
        shape = list(input_shape)
        shape[self.axis] = None
        return tuple(shape)


def apply_mask(layer_seq, layer_seq_mask):
    """
    seq: layer of shape (batch_size, length_seq, n_features)
    seq_mask: layer of shape (batch_size, length_seq)
    """
    return ElemwiseMergeLayer(
        [ForgetSizeLayer(dimshuffle(layer_seq_mask,
                                    (0, 1, 'x'))), layer_seq], T.mul)


def create_deep_rnn(layer, layer_mask, layer_class, depth, residual=False,
                    skip_connections=False, bidir=False, dropout=None,
                    **kwargs):
    layers = [layer]
    for i in range(depth):
        if skip_connections and i > 0:
            layer = concat([layers[0], layer], axis=2)

        new_layer = layer_class(layer, mask_input=layer_mask, **kwargs)

        if bidir:
            layer_bw = layer_class(layer, mask_input=layer_mask,
                                   backwards=True, **kwargs)
            new_layer = concat([new_layer, layer_bw], axis=2)

        if residual:
            layer = ElemwiseSumLayer([layer, new_layer])
        else:
            layer = new_layer

        if skip_connections and i == depth-1:
            layer = concat([layer] + layers[1:], axis=2)

        if dropout:
            layer = DropoutLayer(layer, p=dropout)

        layers.append(layer)

    return layer


def non_flattening_dense_layer(layer, mask, num_units, *args, **kwargs):
    batchsize, seqlen = mask.input_var.shape
    l_flat = ReshapeLayer(layer, (-1, [2]))
    l_dense = DenseLayer(l_flat, num_units, *args, **kwargs)
    return ReshapeLayer(l_dense, (batchsize, seqlen, -1))
