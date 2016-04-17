import theano
import theano.tensor as T
from lasagne.init import GlorotNormal

from raccoon.archi import (GRULayer, EmbeddingLayer,
                           AttentionLayerNaive, FFLayer)

floatX = theano.config.floatX = 'float32'


class Model:
    def __init__(self, vocab_size, embedding_size, n_hidden_quest,
                 n_hidden_cont, n_attention, n_out_hidden,
                 n_entities):

        # input
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.n_hidden_quest = n_hidden_quest
        self.n_hidden_cont = n_hidden_cont

        # attention
        self.n_attention = n_attention

        # output
        self.n_out_hidden = n_out_hidden
        self.n_entities = n_entities

        ini = GlorotNormal()

        # Sequence processing
        self.embedding = EmbeddingLayer(vocab_size, embedding_size, ini)
        self.context_gru = GRULayer(embedding_size, n_hidden_cont, ini)
        self.question_gru = GRULayer(embedding_size, n_hidden_quest, ini)

        # Attention network
        self.att_cont_layer = FFLayer(n_hidden_cont, n_attention, ini)
        self.att_que_layer = FFLayer(n_hidden_quest, n_attention, ini)
        self.att_layer2 = FFLayer(n_attention, 1, ini)
        self.att_weigh_layer = AttentionLayerNaive()

        # Output
        self.out_att_layer = FFLayer(n_hidden_cont, n_out_hidden, ini)
        self.out_que_layer = FFLayer(n_hidden_quest, n_out_hidden, ini)
        self.out_layer = FFLayer(n_out_hidden, n_entities, ini)

        # all parameters
        self.params = (
            self.embedding.params +
            self.context_gru.params +
            self.question_gru.params +

            self.att_cont_layer.params +
            self.att_que_layer.params +
            self.att_layer2.params +

            self.out_att_layer.params +
            self.out_que_layer.params +
            self.out_layer.params)

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

        h_question_ini = T.alloc(0.0, batch_size, self.n_hidden_quest)
        seq_h_quest, updates_quest = self.question_gru.apply(
                seq_question, seq_question_mask, h_question_ini)

        h_context_ini = T.alloc(0.0, batch_size, self.n_hidden_cont)
        seq_h_cont, updates_cont = self.context_gru.apply(
            seq_context, seq_context_mask, h_context_ini)

        #############
        # ATTENTION #
        #############

        # (batch_size, n_hidden_question)
        att_quest = seq_h_quest[-1]

        # (batch_size, n_attention)
        att_quest2 = self.att_que_layer.apply(att_quest)

        # (length_seq_context, batch_size, n_attention)
        seq_cont_att = self.att_cont_layer.apply(seq_h_cont)

        # (length_seq_context, batch_size, n_attention)
        att = T.tanh(seq_cont_att + att_quest2)

        # (length_seq_context, batch_size, 1)
        att = self.att_layer2.apply(att)[:, :, 0]

        # (batch_size, n_hidden_context)
        att = self.att_weigh_layer.apply(seq_h_cont, seq_context_mask, att)

        ##########
        # OUTPUT #
        ##########

        # (batch_size, n_out_hidden)
        out_att_cont = self.out_att_layer.apply(att)

        # (batch_size, n_out_hidden)
        out_quest = self.out_que_layer.apply(att_quest)

        # (batch_size, n_entities)
        output = self.out_layer.apply(T.tanh(out_att_cont + out_quest))

        # Select only the entities that are candidates
        is_candidate = T.eq(
            T.arange(self.n_entities, dtype='int32')[None, None, :],
            T.switch(candidates_mask, candidates,
                     -T.ones_like(candidates))[:, :, None]).sum(axis=1)

        # (batch_size, n_entities)
        output = T.switch(is_candidate, output, -1000 * T.ones_like(output))

        #######################
        # COST AND MONITORING #
        #######################

        # (batch_size,)
        pred = output.argmax(axis=1)

        negll = T.nnet.categorical_crossentropy(T.nnet.softmax(output), tg)
        c = negll.mean()
        c.name = 'negll'

        error_rate = T.neq(tg, pred).mean()
        error_rate.name = 'error_rate'

        return c, updates_cont + updates_quest, [error_rate]
