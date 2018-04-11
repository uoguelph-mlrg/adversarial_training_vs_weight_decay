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

    # CIFAR10-specific dimensions
    img_rows = 32
    img_cols = 32
    channels = 3
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
        preds = model(x)

        saver = tf.train.Saver()

        # Restore the checkpoint
        saver.restore(sess, model_file)

        # Iterate over the samples batch-by-batch
        num_eval_examples = config['num_eval_examples']
        eval_batch_size = config['eval_batch_size']

        eval_par = {'batch_size': eval_batch_size}

        d = img_rows * img_cols * channels
        idx_shuff = np.arange(d)
        np.random.shuffle(idx_shuff)
        N = X_test.shape[0]
        X_test = X_test.reshape(N, d)
        X_test_rand = X_test.copy()
        for i in range(3072):
            X_test_rand[:, idx_shuff[i]] = X_test[:, idx_shuff[d - i - 1::][0]]
            if i % 100 == 0:
                adv_accuracy = model_eval(sess, x, y, preds, X_test_rand.reshape(
                    N, img_rows, img_cols, channels), Y_test, args=eval_par)
                print('%d, %.4f' % (i, adv_accuracy))
                        #print('%.4f' % adv_accuracy)
