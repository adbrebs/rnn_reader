from collections import OrderedDict
import theano
import theano.tensor as T

from lasagne.layers import (
    EmbeddingLayer, GRULayer, DenseLayer, ElemwiseSumLayer, NonlinearityLayer,
    dimshuffle, SliceLayer, ElemwiseMergeLayer, ExpressionLayer, DropoutLayer,
    InputLayer)

from raccoon.layers.lasagne_extras import (
    EfficientAttentionLayer, CandidateOutputLayer, SequenceSoftmax,
    ForgetSizeLayer, create_deep_rnn, non_flattening_dense_layer)

from models import ReaderTwoSeqModel

floatX = theano.config.floatX = 'float32'


class AttentionModel(ReaderTwoSeqModel):
    def __init__(self, vocab_size, n_entities, embedding_size,
                 n_hidden_que, n_hidden_con, n_out_hidden, residual=False,
                 depth_rnn=1, grad_clipping=10, skip_connections=False,
                 bidir=False, dropout=False, **kwargs):
        ReaderTwoSeqModel.__init__(
            self, vocab_size, n_entities, embedding_size, residual, depth_rnn,
            grad_clipping, skip_connections, bidir, dropout)

        self.n_hidden_question = n_hidden_que
        self.n_hidden_context = n_hidden_con
        self.n_out_hidden = n_out_hidden

        ##################
        # SEQ PROCESSING #
        ##################

        embed_con = EmbeddingLayer(self.in_con, vocab_size, embedding_size)
        embed_que = EmbeddingLayer(self.in_que, vocab_size, embedding_size,
                                   W=embed_con.W)

        gru_con = create_deep_rnn(
            embed_con, GRULayer, depth_rnn, layer_mask=self.in_con_mask,
            num_units=n_hidden_con, grad_clipping=grad_clipping,
            residual=residual, skip_connections=skip_connections,
            bidir=bidir)[-1]
        gru_que = create_deep_rnn(
            embed_que, GRULayer, depth_rnn, layer_mask=self.in_que_mask,
            num_units=n_hidden_que, grad_clipping=grad_clipping,
            residual=residual, skip_connections=skip_connections,
            bidir=bidir)[-1]

        #############
        # ATTENTION #
        #############

        que_condition = SliceLayer(gru_que, indices=-1, axis=1)
        batch_size = self.seq_con.shape[0]
        att = self.create_attention(gru_con, self.in_con_mask, que_condition,
                                    batch_size, n_hidden_con, **kwargs)

        ##########
        # OUTPUT #
        ##########

        out_att = DenseLayer(att, n_out_hidden, nonlinearity=None)
        out_que = DenseLayer(que_condition, n_out_hidden, nonlinearity=None)

        out_sum = ElemwiseSumLayer([out_att, out_que])
        if dropout:
            out_sum = DropoutLayer(out_sum, dropout)
        out_tanh = NonlinearityLayer(out_sum, nonlinearity=T.tanh)

        out = DenseLayer(out_tanh, self.n_entities, nonlinearity=None)
        if dropout:
            out = DropoutLayer(out, dropout)

        self.net = CandidateOutputLayer(out, self.in_cand, self.in_cand_mask)

    def create_attention(self, gru_con, in_con_mask, condition, batch_size,
                         n_hidden_con, **kwargs):
        raise NotImplemented


class NoAttentionModel(AttentionModel):
    """
    Two recurrent networks, one for the question, one for the context.
    The last hidden states are used to predict the answer.
    """
    def create_attention(self, gru_con, in_con_mask, condition, batch_size,
                         n_hidden_con, **kwargs):
        return SliceLayer(gru_con, indices=-1, axis=1)


class EfficientAttentionModel(AttentionModel):
    def create_attention(self, gru_con, in_con_mask, condition, batch_size,
                         n_hidden_con, **kwargs):
        return EfficientAttentionLayer(gru_con, self.in_con_mask, condition,
                                       **kwargs)


class EfficientFixedAttention(AttentionModel):
    def create_attention(self, gru_con, in_con_mask, _, batch_size,
                         n_hidden_con, **kwargs):

        # Trick to create a learnable vector of condition (which corresponds
        # to the matrix of the dense layer)
        fixed_condition = InputLayer(
            (None, 1), input_var=T.ones((batch_size, 1)))
        condition = DenseLayer(
            fixed_condition, n_hidden_con, b=None, nonlinearity=None)

        return EfficientAttentionLayer(gru_con, self.in_con_mask, condition)


class SoftmaxAttentionModel(AttentionModel):
    def __init__(self, vocab_size, n_entities, embedding_size,
                 n_hidden_que, n_hidden_con, n_out_hidden, n_attention,
                 depth_rnn=1, grad_clipping=10, skip_connections=False,
                 residual=False, bidir=False, dropout=False):
        self.n_attention = n_attention
        AttentionModel.__init__(
            self, vocab_size, n_entities, embedding_size, n_hidden_que,
            n_hidden_con, n_out_hidden, depth_rnn=depth_rnn,
            grad_clipping=grad_clipping, residual=residual,
            skip_connections=skip_connections, bidir=bidir, dropout=dropout)

    def create_attention(self, gru_con, in_con_mask, condition, batch_size,
                         n_hidden_con, **kwargs):

        # (batch_size, n_attention)
        gru_cond2 = non_flattening_dense_layer(
            gru_con, self.in_con_mask, self.n_attention, nonlinearity=None)
        gru_que2 = DenseLayer(condition, self.n_attention, nonlinearity=None)
        gru_que2 = dimshuffle(gru_que2, (0, 'x', 1))

        att = ElemwiseSumLayer([gru_cond2, gru_que2])
        att = NonlinearityLayer(att, T.tanh)
        att = SliceLayer(non_flattening_dense_layer(
            att, self.in_con_mask, 1, nonlinearity=None), indices=0, axis=2)

        att_softmax = SequenceSoftmax(att, self.in_con_mask)

        rep = ElemwiseMergeLayer(
            [ForgetSizeLayer(dimshuffle(att_softmax, (0, 1, 'x'))),
             gru_con], T.mul)

        return ExpressionLayer(rep, lambda x: T.sum(x, axis=1),
                               lambda s: (s[0],) + s[2:])
