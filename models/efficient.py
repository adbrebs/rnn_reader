import theano
import theano.tensor as T
from lasagne.init import GlorotNormal

from raccoon.archi import (GRULayer, EmbeddingLayer,
                           AttentionLayerEfficient, RnnCovarianceLayer, FFLayer)

floatX = theano.config.floatX = 'float32'


class Model:
    def __init__(self, vocab_size, embedding_size, n_hidden_question,
                 n_hidden_context, n_out_hidden, n_entities):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.n_hidden_question = n_hidden_question
        self.n_hidden_context = n_hidden_context
        self.n_entities = n_entities
        self.n_out_hidden = n_out_hidden

        ini = GlorotNormal()

        self.embedding = EmbeddingLayer(vocab_size, embedding_size, ini)
        self.gru_context = GRULayer(embedding_size, n_hidden_context, ini)
        self.gru_question = GRULayer(embedding_size, n_hidden_question, ini)
        self.attention_layer = AttentionLayerEfficient()

        self.out_att_layer = FFLayer(n_hidden_context, n_out_hidden, ini)
        self.out_que_layer = FFLayer(n_hidden_context, n_out_hidden, ini)
        self.output_layer = FFLayer(n_hidden_context, n_entities, ini)

        self.params = (self.embedding.params + self.gru_context.params +
                       self.gru_question.params + self.output_layer.params)

    def apply(self, seq_context, seq_context_mask,
              seq_question, seq_question_mask,
              tg, candidates, candidates_mask):
        """
        Parameters
        ----------
        seq_context: (length_seq_context, batch_size)
        seq_context_mask: (length_seq_context, batch_size)
        seq_question: (length_seq_question, batch_size)
        seq_question_mask: (length_seq_question, batch_size)
        tg: (lbatch_size,)
        candidates: (batch_size, n_candidates)
        candidates_mask: (batch_size, n_candidates)
        """
        batch_size = seq_context.shape[1]

        ##################
        # SEQ PROCESSING #
        ##################

        # (length_seq_context, batch_size, embedding_size)
        seq_context = self.embedding.apply(seq_context)

        # (length_seq_context, batch_size, embedding_size)
        seq_question = self.embedding.apply(seq_question)

        h_question_ini = T.alloc(0.0, batch_size, self.n_hidden_question)
        seq_h_quest, updates_quest = self.gru_question.apply(
                seq_question, seq_question_mask, h_question_ini)

        h_context_ini = T.alloc(0.0, batch_size, self.n_hidden_context)
        seq_h_context, updates_cont = self.gru_context.apply(
            seq_context, seq_context_mask, h_context_ini)

        #############
        # ATTENTION #
        #############

        # (batch_size, n_hidden_question)
        att_quest = seq_h_quest[-1]

        seq_h_context *= T.shape_padright(seq_context_mask)
        # (batch_size, n_hidden_question, n_hidden_question)
        covariance = (T.shape_padaxis(seq_h_context, 2) *
                      T.shape_padaxis(seq_h_context, 3))
        covariance = T.sum(covariance, axis=0)

        # (batch_size, n_hidden)
        att_cont = self.attention_layer.apply(covariance, att_quest)
        norm2_att = T.sum(att_quest * att_cont, axis=1)
        att_cont /= norm2_att[:, None]

        ##########
        # OUTPUT #
        ##########

        # (batch_size, n_out_hidden)
        out_att_cont = self.out_att_layer.apply(att_cont)

        # (batch_size, n_out_hidden)
        out_quest = self.out_que_layer.apply(att_quest)

        # (batch_size, n_entities)
        output = self.output_layer.apply(T.tanh(out_att_cont + out_quest))

        is_candidate = T.eq(
            T.arange(self.n_entities, dtype='int32')[None, None, :],
            T.switch(candidates_mask, candidates,
                     -T.ones_like(candidates))[:, :, None]).sum(axis=1)

        output = T.switch(is_candidate, output, -1000 * T.ones_like(output))

        negll = T.nnet.categorical_crossentropy(T.nnet.softmax(output), tg)
        c = negll.mean()
        c.name = 'negll'

        error_rate = T.neq(tg, output.argmax(axis=1)).mean()
        error_rate.name = 'error_rate'

        return c, updates_cont + updates_quest, [error_rate]
