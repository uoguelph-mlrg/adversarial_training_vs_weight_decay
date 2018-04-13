from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import keras
from keras import backend

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
from cleverhans.utils_tf import model_train, model_eval

FLAGS = flags.FLAGS

# Softmax temperature scaling
INIT_T = 1.0
ATTACK_T = 0.25

ATTACK_CARLINI_WAGNER_L2 = 0
ATTACK_JSMA = 1
ATTACK_FGSM = 2
ATTACK_MADRYETAL = 3
ATTACK_EAD = 4
ATTACK_BIM = 5
MAX_BATCH_SIZE = 10

EPS_ITER = 2. / 255
DFL_EPS = 8. / 255


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

    # Create TF session and set as Keras backend session
    sess = tf.Session()
    keras.backend.set_session(sess)

    if FLAGS.debug:
        set_log_level(logging.DEBUG)
    else:
        set_log_level(logging.WARNING)  # for running on sharcnet

    # Get CIFAR10 test data
    X_train, Y_train, X_test, Y_test = data_cifar10()

    assert Y_train.shape[1] == 10.
    label_smooth = .1
    Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(
        None, img_rows, img_cols, channels))

    y = tf.placeholder(tf.float32, shape=(None, 10))
    phase = tf.placeholder_with_default(True, shape=(), name="phase")

    logits_scalar = tf.placeholder_with_default(
        INIT_T, shape=(), name="logits_temperature")

    model_path = FLAGS.model_path
    targeted = True if FLAGS.targeted else False
    binary = True if FLAGS.binary else False
    scale = True if FLAGS.scale else False
    test = True if FLAGS.test else False
    learning_rate = FLAGS.learning_rate
    l1 = FLAGS.l1
    l2 = FLAGS.l2
    batch_size = FLAGS.batch_size
    nb_filters = FLAGS.nb_filters
    nb_samples = FLAGS.nb_samples
    nb_epochs = FLAGS.nb_epochs
    delay = FLAGS.delay
    eps = FLAGS.eps
    adv = FLAGS.adv

    save = False
    train_from_scratch = False

    if model_path is not None:
        if os.path.exists(model_path):
            # check for existing model in immediate subfolder
            if any(f.endswith('.meta') for f in os.listdir(model_path)):
                binary, scale, nb_filters, batch_size, learning_rate, nb_epochs, adv = parse_model_settings(
                    model_path)
                train_from_scratch = False
            else:
                model_path = build_model_save_path(
                    model_path, binary, batch_size, nb_filters, learning_rate, nb_epochs, adv, delay, scale)
                print(model_path)
                save = True
                train_from_scratch = True
    else:
        train_from_scratch = True  # train from scratch, but don't save since no path given

    if binary:
        if scale:
            from cnn_models import make_scaled_binary_cnn
            model = make_scaled_binary_cnn(phase, logits_scalar, 'bin_', input_shape=(
                None, img_rows, img_cols, channels), nb_filters=nb_filters)
        else:
            from cnn_models import make_basic_binary_cnn
            model = make_basic_binary_cnn(phase, logits_scalar, 'bin_', input_shape=(
                None, img_rows, img_cols, channels), nb_filters=nb_filters)
    else:
        from cnn_models import make_basic_cnn
        model = make_basic_cnn(phase, logits_scalar, 'fp_', input_shape=(
            None, img_rows, img_cols, channels), nb_filters=nb_filters)
        '''
        from cnn_models import make_bottle_cnn  
        model = make_bottle_cnn(phase, logits_scalar, 'fp_', input_shape=(
            None, img_rows, img_cols, channels), nb_filters=nb_filters)
        '''

    #preds = model(x, reuse=False)
    preds = model(x)
    print("Defined TensorFlow model graph.")

    rng = np.random.RandomState([2017, 8, 30])

    def evaluate():
        # Evaluate the accuracy of the CIFAR10 model on legitimate test
        # examples
        eval_params = {'batch_size': batch_size}
        acc = model_eval(
            sess, x, y, preds, X_test, Y_test, args=eval_params)
        assert X_test.shape[0] == 10000, X_test.shape
        print('Test accuracy on legitimate examples: %0.4f' % acc)

    # Train an CIFAR10 model
    train_params = {
        'binary': binary,
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'l1': l1,
        'l2': l2,
        'learning_rate': learning_rate,
        'loss_name': 'train loss',
        'filename': 'model',
        'reuse_global_step': tf.AUTO_REUSE,
        'train_scope': 'train'
    }

    if train_from_scratch:
        if save:
            train_params.update({'log_dir': model_path})
            if adv and delay > 0:
                train_params.update({'nb_epochs': delay})

         # do clean training for 'nb_epochs' or 'delay' epochs
        if test:
            model_train(sess, x, y, preds, X_train, Y_train, model=model,
                        evaluate=evaluate, args=train_params, save=save, rng=rng)
        else:
            model_train(sess, x, y, preds, X_train, Y_train, model=model,
                        args=train_params, save=save, rng=rng)
    else:
        model_load(sess, model_path)
        print('Restored model from %s' % model_path)

    # Evaluate the accuracy of the CIFAR10 model on legitimate test examples
    eval_params = {'batch_size': batch_size}
    accuracy = model_eval(sess, x, y, preds, X_test, Y_test, args=eval_params)

    print('Test accuracy on legitimate test examples: {0}'.format(accuracy))

    ###########################################################################
    # Build dataset
    ###########################################################################

    if targeted:
        from utils import build_targeted_dataset
        adv_inputs, true_labels, adv_ys = build_targeted_dataset(
            X_test, Y_test, np.arange(nb_samples), nb_classes, img_rows, img_cols, channels)
    else:
        adv_inputs = X_test[:nb_samples]
        true_labels = Y_test[:nb_samples]

        #from skimage.transform import rotate
        # for i in range(adv_inputs.shape[0]):
        #    adv_inputs[i] = rotate(adv_inputs[i], 20, mode='edge')

    ###########################################################################
    # Craft adversarial examples using generic approach
    ###########################################################################
    if targeted:
        att_batch_size = np.clip(
            nb_samples * (nb_classes - 1), a_max=MAX_BATCH_SIZE, a_min=1)
        nb_adv_per_sample = nb_classes - 1
        yname = "y_target"

    else:
        att_batch_size = np.minimum(nb_samples, MAX_BATCH_SIZE)
        nb_adv_per_sample = 1
        adv_ys = None
        yname = "y"

    print('Crafting ' + str(nb_samples) + ' * ' + str(nb_adv_per_sample) +
          ' adversarial examples')
    print("This could take some time ...")

    from cleverhans.attacks import MadryEtAl
    attacker = MadryEtAl(model, back='tf', sess=sess)
    attack_params = {'eps': eps, 'eps_iter': EPS_ITER,
                     'nb_iter': FLAGS.nb_iter, 'ord': np.inf,
                     'rand_init': True
                     }

    attack_params.update({yname: adv_ys, 'clip_min': 0., 'clip_max': 1.})
    X_test_adv = attacker.generate_np(adv_inputs, **attack_params)

    if targeted:
        assert X_test_adv.shape[0] == nb_samples * \
            (nb_classes - 1), X_test_adv.shape
        # Evaluate the accuracy of the CIFAR10 model on adversarial
        # examples
        print("Evaluating targeted results")
        adv_accuracy = model_eval(sess, x, y, preds, X_test_adv, true_labels,
                                  args=eval_params)
    else:
        assert X_test_adv.shape[0] == nb_samples, X_test_adv.shape
        # Evaluate the accuracy of the CIFAR10 model on adversarial
        # examples
        print("Evaluating un-targeted results")
        adv_accuracy = model_eval(sess, x, y, preds, X_test_adv, Y_test,
                                  args=eval_params)

    # Compute the number of adversarial examples that were successfully
    # found
    print('Test accuracy on PGD (20-2-8) examples {0:.4f}'.format(
        adv_accuracy))

    # Compute the average distortion introduced by the algorithm
    diff = np.abs(X_test_adv - adv_inputs)

    percent_perturbed = np.mean(np.sum(diff, axis=(1, 2, 3)))
    print('Avg. L_1 norm of perturbations {0:.4f}'.format(
        percent_perturbed))

    norm = np.mean(np.sqrt(np.sum(np.square(diff), axis=(1, 2, 3))))
    print('Avg. L_2 norm of perturbations {0:.4f}'.format(norm))

    # Friendly output for pasting into spreadsheet
    print('{0:.4f}'.format(accuracy))
    print('{0:.4f}'.format(adv_accuracy))
    print('{0:.4f}'.format(norm))
    print(model.n_params)

    sess.close()


if __name__ == '__main__':

    par = argparse.ArgumentParser()

    # Generic flags
    par.add_argument('--gpu', help='id of GPU to use')
    par.add_argument('--model_path', help='Path to model ckpt',
                     default='./ckpt/model-78199')
    par.add_argument('--data_dir', help='Path to training data',
                     default='/scratch/gallowaa/cifar10/cifar10_data')
    par.add_argument(
        '--debug', help='Sets log level to DEBUG, otherwise INFO', action="store_true")
    par.add_argument(
        '--test', help='test after every epoch', action="store_true")
    par.add_argument(
        '--plot', help="generate unrecognizable examples", action="store_true")
    par.add_argument(
        '--save', help="save weights to fig", action="store_true")

    # Architecture and training specific flags
    par.add_argument('--nb_epochs', type=int, default=15,
                     help='Number of epochs to train model')
    par.add_argument('--nb_filters', type=int, default=32,
                     help='Number of filters in first layer')
    par.add_argument(
        '--l1', help='how much to push weights to zero', type=float, default=0)
    par.add_argument(
        '--l2', help='how much to push weights to zero', type=float, default=0)
    par.add_argument('--batch_size', type=int, default=128,
                     help='Size of training batches')
    par.add_argument('--learning_rate', type=float, default=0.001,
                     help='Learning rate')
    par.add_argument('--binary', help='Use a binary model?',
                     action="store_true")
    par.add_argument('--scale', help='Scale activations of the binary model?',
                     action="store_true")

    # Attack specific flags
    par.add_argument('--eps', type=float, default=DFL_EPS, help='epsilon')
    par.add_argument('--nb_samples', type=int,
                     default=1000, help='Nb of inputs to attack')
    par.add_argument(
        '--targeted', help='Run a targeted attack?', action="store_true")

    # Adversarial training flags
    par.add_argument(
        '--adv', help='Adversarial training type', type=int, default=0)
    par.add_argument('--delay', type=int,
                     default=5, help='Nb of epochs to delay adv training by')
    par.add_argument('--nb_iter', type=int,
                     default=20, help='Nb of iterations of PGD')

    FLAGS = par.parse_args()

    if FLAGS.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    tf.app.run()
