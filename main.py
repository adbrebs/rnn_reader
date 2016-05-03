import argparse
import imp
import os
import shutil
import sys

import numpy as np
import theano

from lasagne.updates import adam, sgd, momentum
from lasagne.layers import get_all_params, get_output

from raccoon.trainer import Trainer
from raccoon.extensions import (TrainMonitor, ValMonitor, VariableSaver,
                                MaxTime, MaxIteration)

from data import create_data_generators
from utilities import save_config

floatX = theano.config.floatX = 'float32'
# theano.config.optimizer = 'None'
# theano.config.compute_test_value = 'raise'
# np.random.seed(42)


def main(cf):

    ########
    # DATA #
    ########

    print 'Creating data generators...'
    train_iterator, valid_iterator, test_iterator = create_data_generators(cf)

    ##############################
    # COST, GRADIENT AND UPDATES #
    ##############################

    print 'Building model...'

    cost, accuracy = cf.model.compute_cost(deterministic=False)
    cost_val, accuracy_val = cf.model.compute_cost(deterministic=True)

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

    print 'Creating extensions and compiling functions...',

    train_monitor = TrainMonitor(
        cf.train_freq_print, cf.model.vars, [cost, accuracy], updates)

    monitoring_vars = [cost_val, accuracy_val]
    valid_monitor = ValMonitor(
        'Validation', cf.valid_freq_print, cf.model.vars, monitoring_vars,
        valid_iterator)

    test_monitor = ValMonitor(
        'Test', cf.valid_freq_print, cf.model.vars, monitoring_vars,
        valid_iterator)

    train_saver = VariableSaver(
        train_monitor, cf.dump_every_batches, cf.dump_path, 'train')

    valid_saver = VariableSaver(
        valid_monitor, cf.dump_every_batches, cf.dump_path, 'valid')

    test_saver = VariableSaver(test_monitor, None, cf.dump_path, 'test')

    # Ending conditions
    end_conditions = []
    if hasattr(cf, 'max_iter'):
        end_conditions.append(MaxIteration(cf.max_iter))
    if hasattr(cf, 'max_time'):
        end_conditions.append(MaxTime(cf.max_iter))

    extensions = [
        valid_monitor,
        test_monitor,

        train_saver,
        valid_saver,
        test_saver
    ]

    train_m = Trainer(train_monitor, train_iterator,
                      extensions, end_conditions)

    ############
    # TRAINING #
    ############

    train_m.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp_name', default='bazar')
    parser.add_argument('-s', '--config_path')
    parser.add_argument('-i', '--job_id', default=None)
    options = parser.parse_args()

    if not options.job_id:
        job_id = np.random.randint(10**6)
        folder = options.config_path + '_' + str(job_id)
    else:
        job_id = options.job_id
        folder = job_id

    print 'EXP name: {}'.format(options.exp_name)
    print 'config file: {}'.format(options.config_path)
    print 'job ID: {}'.format(job_id)

    config = imp.load_source('config', options.config_path)

    config_name = os.path.splitext(options.config_path)[0]
    dump_path = os.path.join(os.getenv('TMP_PATH'), 'QA',
                             options.exp_name, folder)
    if not os.path.exists(dump_path):
        os.makedirs(dump_path)
    config.dump_path = dump_path

    # Copy config file in the dump experiment path
    shutil.copy(options.config_path, os.path.join(dump_path, 'cf.py'))

    # Save config parameters (some of them might be generated at test time)
    print save_config(config, dump_path)

    main(config)
