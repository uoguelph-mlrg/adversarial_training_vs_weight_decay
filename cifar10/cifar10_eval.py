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
from cleverhans.utils import set_log_level, AccuracyReport
from cleverhans.utils_tf import model_eval

FLAGS = flags.FLAGS

EPS_ITER = 2
MAX_BATCH_SIZE = 100


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
    # Build dataset to perturb
    ###########################################################################
    if FLAGS.targeted:
        from utils import build_targeted_dataset
        adv_inputs, true_labels, adv_ys = build_targeted_dataset(
            X_test, Y_test, np.arange(nb_samples), nb_classes, img_rows, img_cols, channels)
        att_batch_size = np.clip(
            nb_samples * (nb_classes - 1), a_max=MAX_BATCH_SIZE, a_min=1)
        nb_adv_per_sample = nb_classes - 1
        yname = "y_target"
    else:
        adv_inputs = X_test[:nb_samples]
        true_labels = Y_test[:nb_samples]
        att_batch_size = np.minimum(nb_samples, MAX_BATCH_SIZE)
        nb_adv_per_sample = 1
        adv_ys = None
        yname = "y"

    print('Crafting ' + str(nb_samples) + ' * ' + str(nb_adv_per_sample) +
          ' adversarial examples')
    print("This could take some time ...")

    if FLAGS.attack == 'pgd':
        from cleverhans.attacks import MadryEtAl
        attacker = MadryEtAl(model, sess=sess)
        attack_params = {'eps': FLAGS.eps / 255., 'eps_iter': EPS_ITER / 255.,
                         'nb_iter': FLAGS.nb_iter, 'ord': np.inf,
                         'rand_init': True, 'batch_size': att_batch_size
                         }
    elif FLAGS.attack == 'cwl2':
        from cleverhans.attacks import CarliniWagnerL2
        attacker = CarliniWagnerL2(model, sess=sess)
        learning_rate = 0.1
        attack_params = {'binary_search_steps': 1,
                         'max_iterations': FLAGS.nb_iter,
                         'learning_rate': learning_rate,
                         'initial_const': 10,
                         'batch_size': att_batch_size
                         }
    attack_params.update({'clip_min': 0.,
                          'clip_max': 1., })
    # yname: adv_ys})

    X_test_adv = attacker.generate_np(adv_inputs, **attack_params)

    if FLAGS.targeted:
        assert X_test_adv.shape[0] == nb_samples * \
            (nb_classes - 1), X_test_adv.shape
        # Evaluate the accuracy of the CIFAR10 model on adversarial
        # examples
        print("Evaluating targeted results")
        # adv_accuracy = model_eval(sess, x, y, preds, X_test_adv, true_labels,
        adv_accuracy = model_eval(sess, x, y, preds_adv, adv_inputs, true_labels,
                                  args=eval_params)
    else:
        # Evaluate the accuracy of the CIFAR10 model on adversarial
        # examples
        print("Evaluating un-targeted results")
        adv_accuracy = model_eval(
            sess, x, y, preds, X_test_adv, Y_test, args=eval_params)

    print('Test accuracy on adversarial examples %.4f' % adv_accuracy)

    # Compute the avg. distortion introduced by the attack
    diff = np.abs(X_test_adv - adv_inputs)

    percent_perturbed = np.mean(np.sum(diff, axis=(1, 2, 3)))
    print('Avg. L_1 norm of perturbations {0:.4f}'.format(
        percent_perturbed))

    norm = np.mean(np.sqrt(np.sum(np.square(diff), axis=(1, 2, 3))))
    print('Avg. L_2 norm of perturbations {0:.4f}'.format(norm))

    sess.close()


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
                     help='Size of evaluation batches')

    par.add_argument("--attack", help='which attack to evaluate with', type=str,
                     default='pgd', choices=['pgd', 'cwl2', 'jsma'])

    par.add_argument('--eps', type=float, default=8, help='epsilon')

    par.add_argument('--nb_samples', type=int,
                     default=1000, help='Nb of inputs to attack')

    par.add_argument(
        '--targeted', help='Run a targeted attack?', action="store_true")

    par.add_argument('--nb_iter', type=int,
                     default=20, help='iterations of PGD')

    FLAGS = par.parse_args()

    if FLAGS.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    tf.app.run()
