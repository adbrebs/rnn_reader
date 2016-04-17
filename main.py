import argparse
import cPickle
import imp
import os
import sys

from lasagne.updates import adam
import numpy as np
import theano
import theano.tensor as T

from raccoon.trainer import Trainer
from raccoon.extensions import TrainMonitor, ValMonitor
from raccoon.archi.utils import clip_norm_gradients

from utilities import create_train_tag_values
from data import create_data_generator

floatX = theano.config.floatX = 'float32'
theano.config.optimizer = 'None'
theano.config.compute_test_value = 'raise'
np.random.seed(42)


def main(cf):

    ########
    # DATA #
    ########

    print 'Create data generators...',
    data_path = os.path.join(os.getenv("DATAPATH"), "deepmind-qa/cnn")
    train_path = os.path.join(data_path, "questions/training")
    valid_path = os.path.join(data_path, "questions/validation")
    vocab_path = os.path.join(data_path, "stats/training/vocab.txt")

    train_iterator = create_data_generator(train_path, vocab_path, cf)
    valid_iterator = create_data_generator(valid_path, vocab_path, cf)
    print 'Done.'

    ##################
    # MODEL CREATION #
    ##################

    seq_cont = T.matrix('seq_cont', 'int32')
    seq_cont_mask = T.matrix('seq_cont_mask', floatX)
    seq_quest = T.matrix('seq_quest', 'int32')
    seq_quest_mask = T.matrix('seq_quest_mask', floatX)
    tg = T.vector('tg', 'int32')
    candidates = T.matrix('candidates', 'int32')
    candidates_mask = T.matrix('candidates_mask', floatX)

    create_train_tag_values(seq_cont, seq_cont_mask, seq_quest, seq_quest_mask,
                            tg, candidates, candidates_mask, cf)

    cost, scan_updates, monitoring = cf.model.apply(
        seq_context=seq_cont,
        seq_context_mask=seq_cont_mask,
        seq_question=seq_quest,
        seq_question_mask=seq_quest_mask,
        tg=tg,
        candidates=candidates,
        candidates_mask=candidates_mask)

    ########################
    # GRADIENT AND UPDATES #
    ########################

    params = cf.model.params
    grads = T.grad(cost, params)
    grads = clip_norm_gradients(grads)

    if cf.algo == 'adam':
        updates_params = adam(grads, params, 0.0003)
    elif cf.algo == 'sgd':
        updates_params = []
        for p, g in zip(params, grads):
            updates_params.append((p, p - cf.learning_rate * g))
    else:
        raise ValueError('Specified algo does not exist')

    updates_all = scan_updates + updates_params

    ##############
    # MONITORING #
    ##############

    train_monitor = TrainMonitor(
        cf.train_freq_print, [seq_cont, seq_cont_mask, seq_quest, seq_quest_mask,
                              tg, candidates, candidates_mask],
        [cost] + monitoring, updates_all)

    valid_monitor = ValMonitor(
        'Validation', cf.valid_freq_print,
        [seq_cont, seq_cont_mask, seq_quest, seq_quest_mask,
         tg, candidates, candidates_mask], [cost] + monitoring,
        valid_iterator, apply_at_the_start=False)

    train_m = Trainer(train_monitor, [valid_monitor], [])

    ############
    # TRAINING #
    ############

    it = epoch = 0

    try:
        while True:
            epoch += 1
            for inputs in train_iterator():
                # print inputs[0].shape
                res = train_m.process_batch(epoch, it, *inputs)

                it += 1
                if res:
                    train_m.finish(it)
                    sys.exit()
    except KeyboardInterrupt:
        print 'Training interrupted by user.'
        train_m.finish(it)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--config_path')
    options = parser.parse_args()

    config = imp.load_source('config', options.config_path)

    main(config)
