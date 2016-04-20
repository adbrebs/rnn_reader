import argparse
import imp
import os
import sys

import numpy as np
import theano

from lasagne.objectives import categorical_accuracy, categorical_crossentropy
from lasagne.updates import adam, sgd, momentum
from lasagne.layers import get_all_params, get_output

from raccoon.trainer import Trainer
from raccoon.extensions import TrainMonitor, ValMonitor, VariableSaver

from data import create_data_generators

floatX = theano.config.floatX = 'float32'
# theano.config.optimizer = 'None'
# theano.config.compute_test_value = 'raise'
# np.random.seed(42)


def main(cf):

    ########
    # DATA #
    ########

    train_iterator, valid_iterator, test_iterator = create_data_generators(cf)

    ##############################
    # COST, GRADIENT AND UPDATES #
    ##############################

    output = get_output(cf.model.net)

    tg = cf.model.tg
    cost = categorical_crossentropy(output, tg).mean()
    cost.name = 'negll'

    accuracy = categorical_accuracy(output, tg).mean()
    accuracy.name = 'accuracy'

    params = get_all_params(cf.model.net, trainable=True)

    if cf.algo == 'adam':
        updates = adam(cost, params, cf.learning_rate)
    elif cf.algo == 'sgd':
        updates = sgd(cost, params, cf.learning_rate)
    elif cf.algo == 'momentum':
        updates = momentum(cost, params, cf.learning_rate)
    else:
        raise ValueError('Specified algo does not exist')

    ##############
    # MONITORING #
    ##############

    monitoring_vars = [cost, accuracy]

    train_monitor = TrainMonitor(
        cf.train_freq_print, cf.model.vars, monitoring_vars, updates)

    valid_monitor = ValMonitor(
        'Validation', cf.valid_freq_print, cf.model.vars, monitoring_vars,
        valid_iterator, apply_at_the_start=False)

    test_monitor = ValMonitor(
        'Test', cf.valid_freq_print, cf.model.vars, monitoring_vars,
        valid_iterator, apply_at_the_end=True)

    train_saver = VariableSaver(
        train_monitor, cf.dump_every_batches, cf.dump_path, 'train')

    valid_saver = VariableSaver(
        valid_monitor, cf.dump_every_batches, cf.dump_path, 'valid')

    test_saver = VariableSaver(test_monitor, None, cf.dump_path, 'test')

    extensions = [
        valid_monitor,
        test_monitor,

        train_saver,
        valid_saver,
        test_saver
    ]

    train_m = Trainer(train_monitor, extensions, [])

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

    exp_id = np.random.randint(10**12)
    dump_path = os.path.join(os.getenv('TMP_PATH'), 'deepmind_qa', str(exp_id))
    if not os.path.exists(dump_path):
        os.makedirs(dump_path)
    config.dump_path = dump_path

    main(config)
