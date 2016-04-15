import theano
from theano import shared
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import numpy as np
from lasagne.init import GlorotNormal

from raccoon.archi import (GRULayer, PositionAttentionLayer, EmbeddingLayer,
                           AttentionLayerEfficient, AttentionLayerNaive,
                           RnnCovarianceLayer, FFLayer)
from raccoon.archi.utils import create_uneven_weight

theano.config.floatX = 'float32'
floatX = theano.config.floatX


class Model1:
    def __init__(self, vocab_size, embedding_size, n_hidden_question,
                 n_hidden_context, n_entities):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.n_hidden_question = n_hidden_question
        self.n_hidden_context = n_hidden_context
        self.n_entities = n_entities

        ini = GlorotNormal()

        self.embedding = EmbeddingLayer(vocab_size, embedding_size, ini)
        self.gru_context = RnnCovarianceLayer(
            GRULayer(embedding_size, n_hidden_context, ini))
        self.gru_question = GRULayer(embedding_size, n_hidden_question, ini)
        self.attention_layer = AttentionLayerEfficient()
        self.output_layer = FFLayer(n_hidden_context, n_entities, ini)

        self.params = (self.embedding.params + self.gru_context.params +
                       self.gru_question.params + self.output_layer.params)

    def apply(self, seq_context, seq_context_mask,
              seq_question, seq_question_mask,
              tg, candidates, candidates_mask,
              h_question_ini, h_context_ini):
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

        # (length_seq_context, batch_size, embedding_size)
        seq_context = self.embedding.apply(seq_context)

        # (length_seq_context, batch_size, embedding_size)
        seq_question = self.embedding.apply(seq_question)

        seq_h_quest, updates_quest = self.gru_question.apply(
                seq_question, seq_question_mask, h_question_ini)

        condition = seq_h_quest[-1]

        (seq_h_cont, covariance_cont), updates_cont = self.gru_question.apply(
            seq_context, seq_context_mask, h_context_ini)

        att = self.attention_layer.apply(covariance_cont, condition)

        output = self.output_layer.apply(att)

        is_candidate = T.eq(
            T.arange(self.n_entities, dtype='int32')[None, None, :],
            T.switch(candidates_mask, candidates,
                     -T.ones_like(candidates))[:, :, None]).sum(axis=1)

        output = T.switch(is_candidate, output, -1000 * T.ones_like(output))

        pred = output.argmax(axis=1)

        negll = T.nnet.categorical_crossentropy(T.nnet.softmax(output), tg)
        c = negll.mean()
        c.name = 'negll'

        error_rate = T.neq(tg, pred).mean()

        return c, updates_cont + updates_quest, [error_rate]
