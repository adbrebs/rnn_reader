import cPickle
import numpy as np
import os
import theano

floatX = theano.config.floatX = 'float32'


def create_train_tag_values(seq_cont, seq_cont_mask, seq_quest, seq_quest_mask,
                            tg, candidates, candidates_mask, config):

    seq_cont_length = 3
    seq_quest_length = 2
    batch_size = config.batch_size
    vocab_size = config.vocab_size
    n_entities = config.n_entities
    n_candidates = 5

    seq_cont.tag.test_value = np.random.randint(0, vocab_size, (seq_cont_length, batch_size)).astype('int32')
    seq_cont_mask.tag.test_value = np.ones((seq_cont_length, batch_size), dtype=floatX)

    seq_quest.tag.test_value = np.random.randint(0, vocab_size, (seq_quest_length, batch_size)).astype('int32')
    seq_quest_mask.tag.test_value = np.ones((seq_quest_length, batch_size), dtype=floatX)

    tg.tag.test_value = np.random.randint(0, n_entities, (batch_size,)).astype('int32')

    candidates.tag.test_value = np.repeat(np.arange(n_candidates, dtype='int32')[None, :], batch_size, axis=0)
    candidates_mask.tag.test_value = np.ones((batch_size, n_candidates), dtype=floatX)


def save_config(cf, dump_path):
    params = {}

    save_field('embedding_size', cf, params)
    save_field('n_hidden_que', cf, params)
    save_field('n_hidden_con', cf, params)
    save_field('depth_rnn', cf, params)
    save_field('grad_clipping', cf, params)
    save_field('residual', cf, params)
    save_field('skip_connections', cf, params)
    save_field('bidir', cf, params)
    save_field('dropout', cf, params)
    save_field('learning_rate', cf, params)

    cPickle.dump(params, open(os.path.join(dump_path, 'cf_params.pkl'), 'wb'))

    return params


def save_field(field_name, cf, params):
    if hasattr(cf, field_name):
        params[field_name] = getattr(cf, field_name)