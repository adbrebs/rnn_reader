import theano
import theano.tensor as T

from lasagne.layers import (EmbeddingLayer, GRULayer, DenseLayer, SliceLayer,
                            DropoutLayer)

from raccoon.layers.lasagne_extras import CandidateOutputLayer, create_deep_rnn

from models import ReaderOneSeqModel, ReaderTwoSeqModel

floatX = theano.config.floatX = 'float32'


class GRUOneSeqModel(ReaderOneSeqModel):
    def __init__(self, vocab_size, n_entities, embedding_size,
                 n_hidden, n_out_hidden, residual=False,
                 depth_rnn=1, grad_clipping=10, skip_connections=False,
                 bidir=False, dropout=False):
        ReaderOneSeqModel.__init__(
            self, vocab_size, n_entities, embedding_size, residual, depth_rnn,
            grad_clipping, skip_connections, bidir, dropout)

        self.n_hidden = n_hidden
        self.n_out_hidden = n_out_hidden

        ##################
        # SEQ PROCESSING #
        ##################

        embed = EmbeddingLayer(self.in_seq, vocab_size, embedding_size)

        rnn = create_deep_rnn(
            embed, GRULayer, depth_rnn, layer_mask=self.in_mask,
            num_units=n_hidden, grad_clipping=grad_clipping,
            residual=residual, skip_connections=skip_connections, bidir=bidir,
            dropout=dropout)[-1]

        rnn_last = SliceLayer(rnn, indices=-1, axis=1)

        ##########
        # OUTPUT #
        ##########

        out = DenseLayer(rnn_last, self.n_out_hidden, nonlinearity=T.tanh)
        if dropout:
            out = DropoutLayer(out, dropout)
        out = DenseLayer(out, self.n_entities, nonlinearity=None)
        if dropout:
            out = DropoutLayer(out, dropout)

        self.net = CandidateOutputLayer(out, self.in_cand, self.in_cand_mask)


# class GRUTwoSeqModel(ReaderTwoSeqModel):
#     def __init__(self, vocab_size, n_entities, embedding_size,
#                  n_hidden, n_out_hidden, residual=False,
#                  depth_rnn=1, grad_clipping=10, skip_connections=False,
#                  bidir=False, dropout=False):
#         ReaderTwoSeqModel.__init__(
#             self, vocab_size, n_entities, embedding_size, residual, depth_rnn,
#             grad_clipping, skip_connections, bidir, dropout)
#
#
#         self.n_hidden = n_hidden
#         self.n_out_hidden = n_out_hidden