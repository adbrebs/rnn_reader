import theano
import theano.tensor as T

from lasagne.layers import (
    EmbeddingLayer, GRULayer, DenseLayer, ElemwiseSumLayer, NonlinearityLayer,
    dimshuffle, SliceLayer, ElemwiseMergeLayer, ExpressionLayer, ReshapeLayer)

from models import (
    ReaderModel, EfficientAttentionLayer, CandidateOutputLayer,
    SequenceSoftmax, ForgetSizeLayer)

floatX = theano.config.floatX = 'float32'


class AttentionModel(ReaderModel):
    def __init__(self, vocab_size, n_entities, embedding_size,
                 n_hidden_que, n_hidden_con, n_out_hidden):
        ReaderModel.__init__(self, vocab_size, n_entities, embedding_size)

        self.n_hidden_question = n_hidden_que
        self.n_hidden_context = n_hidden_con
        self.n_out_hidden = n_out_hidden

        ##################
        # SEQ PROCESSING #
        ##################

        embed_con = EmbeddingLayer(self.in_con, vocab_size, embedding_size)
        embed_que = EmbeddingLayer(self.in_que, vocab_size, embedding_size,
                                   W=embed_con.W)

        gru_con = GRULayer(embed_con, n_hidden_con,
                           mask_input=self.in_con_mask, grad_clipping=10)

        gru_que = GRULayer(embed_que, n_hidden_que, grad_clipping=10,
                           mask_input=self.in_que_mask, only_return_final=True)

        #############
        # ATTENTION #
        #############

        att = self.process_attention(gru_con, self.in_con_mask, gru_que)

        ##########
        # OUTPUT #
        ##########

        out_att = DenseLayer(att, n_out_hidden, nonlinearity=None)
        out_que = DenseLayer(gru_que, n_out_hidden, nonlinearity=None)

        out_sum = ElemwiseSumLayer([out_att, out_que])
        out_tanh = NonlinearityLayer(out_sum, nonlinearity=T.tanh)

        out = DenseLayer(out_tanh, self.n_entities, nonlinearity=None)

        self.net = CandidateOutputLayer(out, self.in_cand, self.in_cand_mask)

    def process_attention(self, gru_con, in_con_mask, gru_que):
        raise NotImplemented


class EfficientAttentionModel(AttentionModel):
    def process_attention(self, gru_con, in_con_mask, gru_que):
        return EfficientAttentionLayer(gru_con, self.in_con_mask, gru_que)


class SoftmaxAttentionModel(AttentionModel):
    def __init__(self, vocab_size, n_entities, embedding_size,
                 n_hidden_que, n_hidden_con, n_out_hidden, n_attention):
        self.n_attention = n_attention
        AttentionModel.__init__(self, vocab_size, n_entities, embedding_size,
                                n_hidden_que, n_hidden_con, n_out_hidden)

    def non_flattening_dense(self, layer, *args, **kwargs):
        batchsize, seqlen = self.in_con.input_var.shape
        l_flat = ReshapeLayer(layer, (-1, [2]))
        l_dense = DenseLayer(l_flat, *args, **kwargs)
        return ReshapeLayer(l_dense, (batchsize, seqlen, -1))

    def process_attention(self, gru_con, in_con_mask, gru_que):

        # (batch_size, n_attention)
        gru_cond2 = self.non_flattening_dense(gru_con, self.n_attention,
                                              nonlinearity=None)
        gru_que2 = DenseLayer(gru_que, self.n_attention, nonlinearity=None)
        gru_que2 = dimshuffle(gru_que2, (0, 'x', 1))

        att = ElemwiseSumLayer([gru_cond2, gru_que2])
        att = NonlinearityLayer(att, T.tanh)
        att = SliceLayer(self.non_flattening_dense(att, 1, nonlinearity=None),
                         indices=0, axis=2)

        att_softmax = SequenceSoftmax(att, self.in_con_mask)

        rep = ElemwiseMergeLayer(
            [ForgetSizeLayer(dimshuffle(att_softmax, (0, 1, 'x'))),
             gru_con], T.mul)

        return ExpressionLayer(rep, lambda x: T.sum(x, axis=1),
                               lambda s: (s[0],) + s[2:])
