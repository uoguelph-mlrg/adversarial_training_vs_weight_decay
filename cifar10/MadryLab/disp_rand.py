"""
Running this file as a program will display
CIFAR-10 data samples with random pixels swapped.
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

import matplotlib.pyplot as plt

if __name__ == '__main__':
    import time
    import json
    import sys
    import math

    with open('config.json') as config_file:
        config = json.load(config_file)
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

    d = img_rows * img_cols * channels
    idx_shuff = np.arange(d)
    np.random.shuffle(idx_shuff)
    N = X_test.shape[0]
    X_test = X_test.reshape(N, d)
    X_test_rand = X_test.copy()

    # build a rectangle in axes coords
    left, width = .25, .5
    bottom, height = .25, .5
    right = left + width
    top = bottom + height
    fig, axes = plt.subplots(2, 10, squeeze=True, figsize=(8, 2.2))

    j = 0
    for i in range(1100):
        X_test_rand[:, idx_shuff[i]] = X_test[:, idx_shuff[d - i - 1::][0]]
        if i % 100 == 0 and i > 0:

            axes[0, j].imshow(X_test_rand[j].reshape(32, 32, 3))
            axes[0, j].get_xaxis().set_visible(False)
            axes[0, j].get_yaxis().set_visible(False)

            axes[1, j].imshow(X_test[j].reshape(32, 32, 3))
            axes[1, j].get_xaxis().set_visible(False)
            axes[1, j].get_yaxis().set_visible(False)

            ax = axes[0, j]
            ax.text(0.5 * (left + right), 1.0, "%.3f" % float(i / d), horizontalalignment='center',
                    verticalalignment='bottom', rotation=30, transform=ax.transAxes, size='larger')
            j += 1

    plt.subplots_adjust(left=0.005, bottom=0.005, right=0.995,
                        top=0.7, wspace=0.2, hspace=0.2)
    plt.show()
