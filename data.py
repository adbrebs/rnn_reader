"""
Code from https://github.com/thomasmesnard/DeepMind-Teaching-Machines-to-Read-
and-Comprehend.
Credits: Thomas Mesnard, Etienne Simon, Alex Auvolat.
"""

import logging
import random
import numpy as np
import shutil

from picklable_itertools import iter_

from fuel.datasets import Dataset
from fuel.streams import DataStream
from fuel.schemes import IterationScheme, ConstantScheme
from fuel.transformers import Batch, Mapping, SortMapping, Unpack, Padding, \
    Transformer

import os

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

class QADataset(Dataset):
    def __init__(self, path, vocab_file, n_entities, need_sep_token, **kwargs):
        self.provides_sources = ('context', 'question', 'answer', 'candidates')

        self.path = path

        self.vocab = ['@entity%d' % i for i in range(n_entities)] + \
                     [w.rstrip('\n') for w in open(vocab_file)] + \
                     ['<UNK>', '@placeholder'] + \
                     (['<SEP>'] if need_sep_token else [])

        self.n_entities = n_entities
        self.vocab_size = len(self.vocab)
        self.reverse_vocab = {w: i for i, w in enumerate(self.vocab)}

        super(QADataset, self).__init__(**kwargs)

    def to_word_id(self, w, cand_mapping):
        if w in cand_mapping:
            return cand_mapping[w]
        elif w[:7] == '@entity':
            raise ValueError("Unmapped entity token: %s"%w)
        elif w in self.reverse_vocab:
            return self.reverse_vocab[w]
        else:
            return self.reverse_vocab['<UNK>']

    def to_word_ids(self, s, cand_mapping):
        return np.array([self.to_word_id(x, cand_mapping)
                         for x in s.split(' ')], dtype=np.int32)

    def get_data(self, state=None, request=None):
        if request is None or state is not None:
            raise ValueError("Expected a request (name of a question file) and"
                             " no state.")

        lines = [l.rstrip('\n') for l in open(os.path.join(self.path, request))]

        ctx = lines[2]
        q = lines[4]
        a = lines[6]
        cand = [s.split(':')[0] for s in lines[8:]]

        entities = range(self.n_entities)
        while len(cand) > len(entities):
            logger.warning("Too many entities (%d) for question: %s, using "
                           "duplicate entity identifiers"
                           %(len(cand), request))
            entities = entities + entities
        random.shuffle(entities)
        cand_mapping = {t: k for t, k in zip(cand, entities)}

        ctx = self.to_word_ids(ctx, cand_mapping)
        q = self.to_word_ids(q, cand_mapping)
        cand = np.array([self.to_word_id(x, cand_mapping) for x in cand],
                        dtype=np.int32)
        a = np.int32(self.to_word_id(a, cand_mapping))

        if not a < self.n_entities:
            raise ValueError("Invalid answer token %d"%a)
        if not np.all(cand < self.n_entities):
            raise ValueError("Invalid candidate in list %s"%repr(cand))
        if not np.all(ctx < self.vocab_size):
            raise ValueError("Context word id out of bounds: %d"%int(ctx.max()))
        if not np.all(ctx >= 0):
            raise ValueError("Context word id negative: %d"%int(ctx.min()))
        if not np.all(q < self.vocab_size):
            raise ValueError("Question word id out of bounds: %d"%int(q.max()))
        if not np.all(q >= 0):
            raise ValueError("Question word id negative: %d"%int(q.min()))

        return (ctx, q, a, cand)

class QAIterator(IterationScheme):
    requests_examples = True
    def __init__(self, path, shuffle=False, **kwargs):
        self.path = path
        self.shuffle = shuffle

        super(QAIterator, self).__init__(**kwargs)

    def get_request_iterator(self):
        l = [f for f in os.listdir(self.path)
             if os.path.isfile(os.path.join(self.path, f))]
        if self.shuffle:
            random.shuffle(l)
        return iter_(l)

# -------------- DATASTREAM SETUP --------------------


class ConcatCtxAndQuestion(Transformer):
    produces_examples = True
    def __init__(self, stream, concat_question_before, separator_token=None, **kwargs):
        assert stream.sources == ('context', 'question', 'answer', 'candidates')
        self.sources = ('question', 'answer', 'candidates')

        self.sep = np.array([separator_token] if separator_token is not None else [],
                            dtype=np.int32)
        self.concat_question_before = concat_question_before

        super(ConcatCtxAndQuestion, self).__init__(stream, **kwargs)

    def get_data(self, request=None):
        if request is not None:
            raise ValueError('Unsupported: request')

        ctx, q, a, cand = next(self.child_epoch_iterator)

        if self.concat_question_before:
            return (np.concatenate([q, self.sep, ctx]), a, cand)
        else:
            return (np.concatenate([ctx, self.sep, q]), a, cand)

class _balanced_batch_helper(object):
    def __init__(self, key):
        self.key = key
    def __call__(self, data):
        return data[self.key].shape[0]

def create_data_generator(path, vocab_file, config):
    ds = QADataset(path, vocab_file, config.n_entities, need_sep_token=config.concat_ctx_and_question)
    it = QAIterator(path, shuffle=config.shuffle_questions)

    stream = DataStream(ds, iteration_scheme=it)

    if config.concat_ctx_and_question:
        stream = ConcatCtxAndQuestion(stream, config.concat_question_before, ds.reverse_vocab['<SEP>'])

    # Sort sets of multiple batches to make batches of similar sizes
    stream = Batch(stream, iteration_scheme=ConstantScheme(config.batch_size * config.sort_batch_count))
    comparison = _balanced_batch_helper(stream.sources.index('question' if config.concat_ctx_and_question else 'context'))
    stream = Mapping(stream, SortMapping(comparison))
    stream = Unpack(stream)

    stream = Batch(stream, iteration_scheme=ConstantScheme(config.batch_size))
    stream = Padding(stream, mask_sources=['context', 'question', 'candidates'], mask_dtype='int32')

    def gen():

        if not config.concat_ctx_and_question:
            for (seq_cont, seq_cont_mask, seq_quest, seq_quest_mask, tg,
                 candidates, candidates_mask) in stream.get_epoch_iterator():
                seq_cont_mask = seq_cont_mask.astype('float32')
                seq_quest_mask = seq_quest_mask.astype('float32')
                candidates_mask = candidates_mask.astype('float32')

                yield (seq_cont, seq_cont_mask, seq_quest, seq_quest_mask,
                       tg, candidates, candidates_mask)
        else:

            for (seq, seq_mask, tg, candidates, candidates_mask) \
                    in stream.get_epoch_iterator():
                seq_mask = seq_mask.astype('float32')
                candidates_mask = candidates_mask.astype('float32')

                yield (seq, seq_mask, tg, candidates, candidates_mask)
    return gen


def create_data_generators(cf):
    data_path = os.path.join(os.getenv("TMP_PATH"), "deepmind-qa/cnn")

    if not os.path.exists(data_path):
        original_data_path = os.path.join(os.getenv("DATA_PATH"), "deepmind-qa")
        print '  Dumping data in local folder...',
        shutil.copytree(original_data_path, data_path)
        print ' dumping finished.'

    train_path = os.path.join(data_path, "questions/training")
    valid_path = os.path.join(data_path, "questions/validation")
    test_path = os.path.join(data_path, "questions/test")
    vocab_path = os.path.join(data_path, "stats/training/vocab.txt")

    train_iterator = create_data_generator(train_path, vocab_path, cf)
    valid_iterator = create_data_generator(valid_path, vocab_path, cf)
    test_iterator = create_data_generator(test_path, vocab_path, cf)

    print '  data generators created.'
    return train_iterator, valid_iterator, test_iterator

if __name__ == "__main__":
    # Test
    class DummyConfig:
        def __init__(self):
            self.shuffle_entities = True
            self.shuffle_questions = False
            self.concat_ctx_and_question = False
            self.concat_question_before = False
            self.batch_size = 1
            self.sort_batch_count = 1000
            self.n_entities = 550

    ds, stream = create_data_generator(os.path.join(os.getenv("DATAPATH"), "deepmind-qa/cnn/questions/training"),
                                       os.path.join(os.getenv("DATAPATH"), "deepmind-qa/cnn/stats/training/vocab.txt"),
                                       DummyConfig())
    it = stream.get_epoch_iterator()

    for i, d in enumerate(stream.get_epoch_iterator()):
        print '--'
        for a in d:
            print a.shape
        print '--------------'
        print d
        if i > 1: break

# vim: set sts=4 ts=4 sw=4 tw=0 et :
