from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse
import logging
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.python.platform import flags

from utils import (parse_model_settings,
                   build_model_save_path,
                   model_load, data_cifar10)
from cleverhans.utils import (other_classes, pair_visual,
                              grid_visual, set_log_level,
                              AccuracyReport)
from cleverhans.utils_tf import (model_eval, model_argmax)

FLAGS = flags.FLAGS


def main(argv=None):
    """
    CIFAR10 CleverHans tutorial
    :return:
    """
    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()

    # CIFAR10-specific dimensions
    img_rows = 32
    img_cols = 32
    channels = 3
    nb_classes = 10

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    sess = tf.Session()

    set_log_level(logging.DEBUG)

    # Get CIFAR10 test data
    X_train, Y_train, X_test, Y_test = data_cifar10()

    # Label smoothing
    assert Y_train.shape[1] == 10.

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(
        None, img_rows, img_cols, channels))

    y = tf.placeholder(tf.float32, shape=(None, 10))

    model_path = FLAGS.model_path
    nb_samples = FLAGS.nb_samples

    from cnn_models import make_basic_cnn
    model = make_basic_cnn('fp_', input_shape=(
        None, img_rows, img_cols, channels), nb_filters=FLAGS.nb_filters)

    preds = model(x)
    print("Defined TensorFlow model graph with %d parameters" % model.n_params)

    rng = np.random.RandomState([2017, 8, 30])

    def evaluate(eval_params):
        # Evaluate the model on legitimate test examples
        acc = model_eval(
            sess, x, y, preds, X_test, Y_test, args=eval_params)
        return acc

    model_load(sess, model_path)
    print('Restored model from %s' % model_path)
    eval_params = {'batch_size': FLAGS.batch_size}
    accuracy = evaluate(eval_params)
    print('Test accuracy on legitimate test examples: {0}'.format(accuracy))

    ###########################################################################
    # Craft adversarial examples using the Jacobian-based saliency map approach
    ###########################################################################
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
                   'theta': 1.,
                   'symbolic_impl': True,
                   'clip_min': 0.,
                   'clip_max': 1.,
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

        # For the grid visualization, keep original images along the diagonal
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
            nb_changed = np.where(adv_x_reshape != test_in_reshape)[0].shape[0]
            percent_perturb = float(nb_changed) / adv_x.reshape(-1).shape[0]

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

    # Compute the number of adversarial examples that were successfully found
    nb_targets_tried = ((nb_classes - 1) * nb_samples)
    succ_rate = float(np.sum(results)) / nb_targets_tried
    print('Avg. rate of successful adv. examples {0:.4f}'.format(succ_rate))
    report.clean_train_adv_eval = 1. - succ_rate

    # Compute the average distortion introduced by the algorithm
    percent_perturbed = np.mean(perturbations)
    print('Avg. rate of perturbed features {0:.4f}'.format(percent_perturbed))

    # Compute the average distortion introduced for successful samples only
    percent_perturb_succ = np.mean(perturbations * (results == 1))
    print('Avg. rate of perturbed features for successful '
          'adversarial examples {0:.4f}'.format(percent_perturb_succ))

    # Close TF session
    sess.close()

    # Finally, block & display a grid of all the adversarial examples
    if FLAGS.viz_enabled:
        import matplotlib.pyplot as plt
        plt.close(figure)
        _ = grid_visual(grid_viz_data)


if __name__ == '__main__':

    par = argparse.ArgumentParser()

    par.add_argument('--gpu', help='id of GPU to use')

    par.add_argument('--model_path', help='Path to model ckpt',
                     default='./ckpt/cnn-l2-model')

    par.add_argument('--data_dir', help='Path to CIFAR-10 data',
                     default='/scratch/gallowaa/cifar10/cifar10_data')

    par.add_argument('--nb_filters', type=int, default=32,
                     help='Number of filters in first layer')

    par.add_argument('--batch_size', type=int, default=100,
                     help='batch_size for clean test eval')

    par.add_argument('--gamma', type=float, default=0.05,
                     help='ratio of pixels to perturb')

    par.add_argument('--nb_samples', type=int,
                     default=10, help='Nb of inputs to attack')

    par.add_argument('--viz_enabled', help='visualize result?',
                     action="store_true")

    FLAGS = par.parse_args()

    if FLAGS.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    tf.app.run()
