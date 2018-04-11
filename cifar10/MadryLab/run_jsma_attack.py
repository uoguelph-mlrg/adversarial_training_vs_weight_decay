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

import os
import logging
import argparse
from keras.utils import np_utils
from cleverhans.utils import (other_classes, pair_visual,
                              grid_visual, set_log_level,
                              AccuracyReport)
from cleverhans.utils_tf import (model_eval, model_argmax)
from cleverhans.model import CallableModelWrapper
import cifar10_input

if __name__ == '__main__':

    par = argparse.ArgumentParser()

    par.add_argument('--gpu', help='id of GPU to use')

    par.add_argument('--gamma', type=float, default=0.01,
                     help='ratio of pixels to perturb')

    par.add_argument('--nb_samples', type=int,
                     default=10, help='Nb of inputs to attack')

    par.add_argument('--viz_enabled', help='visualize result?',
                     action="store_true")

    FLAGS = par.parse_args()

    if FLAGS.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

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

    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()

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

        nb_samples = FLAGS.nb_samples
        #######################################################################
        # Craft adversarial examples using the Jacobian-based saliency map approach
        #######################################################################
        print('Crafting ' + str(nb_samples) + ' * ' + str(nb_classes - 1) +
              ' adversarial examples')

        # Keep track of success (adversarial example classified in target)
        results = np.zeros((nb_classes, nb_samples), dtype='i')

        # Rate of perturbed features for each test set example and target class
        perturbations = np.zeros((nb_classes, nb_samples), dtype='f')

        # Initialize our array for grid visualization
        grid_shape = (nb_classes, nb_classes, img_rows, img_cols, channels)
        grid_viz_data = np.zeros(grid_shape, dtype='f')

        from cleverhans.attacks import SaliencyMapMethod
        jsma = SaliencyMapMethod(model, sess=sess)
        jsma_params = {'gamma': FLAGS.gamma,
                       'theta': 255.,
                       'symbolic_impl': True,
                       'clip_min': 0.,
                       'clip_max': 255.,
                       'y_target': None}
        figure = None
        # Loop over the samples we want to perturb into adversarial examples
        for sample_ind in range(0, nb_samples):
            print('--------------------------------------')
            print('Attacking input %i/%i' % (sample_ind + 1, nb_samples))
            sample = X_test[sample_ind:(sample_ind + 1)]

            # We want to find an adversarial example for each possible target class
            # (i.e. all classes that differ from the label given in the dataset)
            current_class = int(np.argmax(Y_test[sample_ind]))
            target_classes = other_classes(nb_classes, current_class)

            # For the grid visualization, keep original images along the
            # diagonal
            grid_viz_data[current_class, current_class, :, :, :] = np.reshape(
                sample, (img_rows, img_cols, channels))

            # Loop over all target classes
            for target in target_classes:
                print('Generating adv. example for target class %i' % target)

                # This call runs the Jacobian-based saliency map approach
                one_hot_target = np.zeros((1, nb_classes), dtype=np.float32)
                one_hot_target[0, target] = 1
                jsma_params['y_target'] = one_hot_target
                adv_x = jsma.generate_np(sample, **jsma_params)

                # Check if success was achieved
                res = int(model_argmax(sess, x, preds, adv_x) == target)

                # Computer number of modified features
                adv_x_reshape = adv_x.reshape(-1)
                test_in_reshape = X_test[sample_ind].reshape(-1)
                nb_changed = np.where(adv_x_reshape != test_in_reshape)[
                    0].shape[0]
                percent_perturb = float(nb_changed) / \
                    adv_x.reshape(-1).shape[0]

                # Display the original and adversarial images side-by-side
                if FLAGS.viz_enabled:
                    figure = pair_visual(
                        np.reshape(sample, (img_rows, img_cols, channels)),
                        np.reshape(adv_x, (img_rows, img_cols, channels)), figure)

                # Add our adversarial example to our grid data
                grid_viz_data[target, current_class, :, :, :] = np.reshape(
                    adv_x, (img_rows, img_cols, channels))

                # Update the arrays for later analysis
                results[target, sample_ind] = res
                perturbations[target, sample_ind] = percent_perturb

        print('--------------------------------------')

        # Compute the number of adversarial examples that were successfully
        # found
        nb_targets_tried = ((nb_classes - 1) * nb_samples)
        succ_rate = float(np.sum(results)) / nb_targets_tried
        print(
            'Avg. rate of successful adv. examples {0:.4f}'.format(succ_rate))
        report.clean_train_adv_eval = 1. - succ_rate

        # Compute the average distortion introduced by the algorithm
        percent_perturbed = np.mean(perturbations)
        print('Avg. rate of perturbed features {0:.4f}'.format(
            percent_perturbed))

        # Compute the average distortion introduced for successful samples only
        percent_perturb_succ = np.mean(perturbations * (results == 1))
        print('Avg. rate of perturbed features for successful '
              'adversarial examples {0:.4f}'.format(percent_perturb_succ))

        # Finally, block & display a grid of all the adversarial examples
        if FLAGS.viz_enabled:
            import matplotlib.pyplot as plt
            plt.close(figure)
            _ = grid_visual(grid_viz_data)
