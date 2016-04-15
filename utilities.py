import numpy as np
import theano
import theano.tensor as T

floatX = theano.config.floatX = 'float32'


def create_train_tag_values(seq_cont, seq_cont_mask, seq_quest, seq_quest_mask,
                            tg, candidates, candidates_mask):

    seq_cont_length = 3
    seq_quest_length = 2
    batch_size = 4
    vocab_size = 20
    n_entities = 10
    n_candidates = 5

    seq_cont.tag.test_value = np.random.randint(0, vocab_size, (seq_cont_length, batch_size)).astype('int32')
    seq_cont_mask.tag.test_value = np.ones((seq_cont_length, batch_size), dtype=floatX)

    seq_quest.tag.test_value = np.random.randint(0, vocab_size, (seq_quest_length, batch_size)).astype('int32')
    seq_quest_mask.tag.test_value = np.ones((seq_quest_length, batch_size), dtype=floatX)

    tg.tag.test_value = np.random.randint(0, n_entities, (batch_size,)).astype('int32')

    candidates.tag.test_value = np.repeat(np.arange(n_candidates, dtype='int32')[None, :], batch_size, axis=0)
    candidates_mask.tag.test_value = np.ones((batch_size, n_candidates), dtype=floatX)