import theano
import theano.tensor as T

from lasagne.layers import (
    EmbeddingLayer, GRULayer, DenseLayer, ElemwiseSumLayer, NonlinearityLayer,
    dimshuffle, SliceLayer, ElemwiseMergeLayer, ExpressionLayer)

from models import (
    ReaderTwoSeqModel, EfficientAttentionLayer, CandidateOutputLayer,
    SequenceSoftmax, ForgetSizeLayer, create_deep_rnn,
    non_flattening_dense_layer)

floatX = theano.config.floatX = 'float32'


class AttentionModel(ReaderTwoSeqModel):
    def __init__(self, vocab_size, n_entities, embedding_size,
                 n_hidden_que, n_hidden_con, n_out_hidden, residual=False,
                 depth_rnn=1, grad_clipping=10, skip_connections=False,
                 bidir=False):
        ReaderTwoSeqModel.__init__(
            self, vocab_size, n_entities, embedding_size, residual, depth_rnn,
            grad_clipping, skip_connections, bidir)

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
            embed_con, self.in_con_mask, GRULayer, depth_rnn,
            num_units=n_hidden_con, grad_clipping=grad_clipping,
            residual=residual, skip_connections=skip_connections, bidir=bidir)
        gru_que = create_deep_rnn(
            embed_que, self.in_que_mask, GRULayer, depth_rnn,
            num_units=n_hidden_que, grad_clipping=grad_clipping,
            residual=residual, skip_connections=skip_connections, bidir=bidir)

        #############
        # ATTENTION #
        #############

        que_condition = SliceLayer(gru_que, indices=-1, axis=1)
        att = self.create_attention(gru_con, self.in_con_mask, que_condition)

        ##########
        # OUTPUT #
        ##########

        out_att = DenseLayer(att, n_out_hidden, nonlinearity=None)
        out_que = DenseLayer(que_condition, n_out_hidden, nonlinearity=None)

        out_sum = ElemwiseSumLayer([out_att, out_que])
        out_tanh = NonlinearityLayer(out_sum, nonlinearity=T.tanh)

        out = DenseLayer(out_tanh, self.n_entities, nonlinearity=None)

        self.net = CandidateOutputLayer(out, self.in_cand, self.in_cand_mask)

    def create_attention(self, gru_con, in_con_mask, condition):
        raise NotImplemented


class EfficientAttentionModel(AttentionModel):
    def create_attention(self, gru_con, in_con_mask, condition):
        return EfficientAttentionLayer(gru_con, self.in_con_mask, condition)


class SoftmaxAttentionModel(AttentionModel):
    def __init__(self, vocab_size, n_entities, embedding_size,
                 n_hidden_que, n_hidden_con, n_out_hidden, n_attention,
                 depth_rnn=1, grad_clipping=10, skip_connections=False,
                 residual=False, bidir=False):
        self.n_attention = n_attention
        AttentionModel.__init__(
            self, vocab_size, n_entities, embedding_size, n_hidden_que,
            n_hidden_con, n_out_hidden, depth_rnn=depth_rnn,
            grad_clipping=grad_clipping, residual=residual,
            skip_connections=skip_connections, bidir=bidir)

    def create_attention(self, gru_con, in_con_mask, condition):

        # (batch_size, n_attention)
        gru_cond2 = non_flattening_dense_layer(gru_con, self.n_attention,
                                               nonlinearity=None)
        gru_que2 = DenseLayer(condition, self.n_attention, nonlinearity=None)
        gru_que2 = dimshuffle(gru_que2, (0, 'x', 1))

        att = ElemwiseSumLayer([gru_cond2, gru_que2])
        att = NonlinearityLayer(att, T.tanh)
        att = SliceLayer(non_flattening_dense_layer(att, 1, nonlinearity=None),
                         indices=0, axis=2)

        att_softmax = SequenceSoftmax(att, self.in_con_mask)

        rep = ElemwiseMergeLayer(
            [ForgetSizeLayer(dimshuffle(att_softmax, (0, 1, 'x'))),
             gru_con], T.mul)

        return ExpressionLayer(rep, lambda x: T.sum(x, axis=1),
                               lambda s: (s[0],) + s[2:])
