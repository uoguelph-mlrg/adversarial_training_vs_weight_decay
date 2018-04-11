"""
Running this file as a program will apply an arbitrary CleverHans
attack to the model specified by the config file.
CleverHans attacks available separately from
https://github.com/tensorflow/cleverhans and must be in PYTHONPATH.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import logging
from keras.utils import np_utils
from cleverhans.utils import set_log_level
from cleverhans.utils_tf import model_eval
from cleverhans.model import CallableModelWrapper
import cifar10_input


if __name__ == '__main__':
    import time
    import json
    import sys
    import math

    with open('config.json') as config_file:
        config = json.load(config_file)

    model_file = tf.train.latest_checkpoint(config['model_dir'])
    if model_file is None:
        print('No model found')
        sys.exit()

    data_path = config['data_path']
    cifar = cifar10_input.CIFAR10Data(data_path)

    nb_classes = 10
    X_test = cifar.eval_data.xs
    Y_test = np_utils.to_categorical(cifar.eval_data.ys, nb_classes)
    assert Y_test.shape[1] == 10.

    set_log_level(logging.DEBUG)

    with tf.Session() as sess:

        x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
        y = tf.placeholder(tf.float32, shape=(None, 10))

        from cleverhans_model import make_madry_wresnet
        model = make_madry_wresnet()

        saver = tf.train.Saver()

        # Restore the checkpoint
        saver.restore(sess, model_file)

        # Iterate over the samples batch-by-batch
        num_eval_examples = config['num_eval_examples']
        eval_batch_size = config['eval_batch_size']

        nb_samples = 10000

        attack_params = {'batch_size': eval_batch_size,
                         'eps': config['epsilon'],
                         'eps_iter': config['step_size'],
                         'nb_iter': config['num_steps'],
                         'clip_min': 0., 'clip_max': 255.,
                         'ord': np.inf}

        #from cleverhans.attacks import FastGradientMethod
        #attacker = FastGradientMethod(model, back='tf', sess=sess)

        from cleverhans.attacks import MadryEtAl
        attacker = MadryEtAl(model, back='tf', sess=sess)

        max_eps = 16
        epsilons = np.linspace(1, max_eps, max_eps)
        #epsilons = np.linspace(0, max_eps, max_eps // 4, endpoint=False)
        eval_par = {'batch_size': eval_batch_size}
        for e in epsilons:
            start_time = time.time()
            attack_params.update({'eps': e})
            x_adv = attacker.generate(x, **attack_params)
            preds_adv = model.get_probs(x_adv)
            acc = model_eval(sess, x, y, preds_adv, X_test[
                :nb_samples], Y_test[:nb_samples], args=eval_par)
            print('%.2f, %.4f, %f' % (e, acc, time.time() - start_time))
        
        '''
        # attacker can be any CleverHans attack here,
        # but some attacks creating internal variables not
        # yet tested.
        attack_params = {'batch_size': eval_batch_size,
                         'clip_min': 0., 'clip_max': 255.}
        attacker = ElasticNetMethod(model, back='tf', sess=sess)
        x_adv = attacker.generate(x, **attack_params)
        preds_adv = model.get_probs(x_adv)
        acc = model_eval(sess, x, y, preds_adv, X_test[
            :nb_samples], Y_test[:nb_samples], args=eval_par)
        print('EAD %.4f' % acc)
        '''
