from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import keras
from keras import backend
from keras.datasets import cifar10
from keras.utils import np_utils

import os
import argparse
import logging
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.python.platform import flags

from utils import parse_model_settings, build_model_save_path, model_load
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
MAX_EPS = 0.3
BN_TEST_PHASE = False


def data_cifar10():
    """
    Preprocess CIFAR10 dataset
    :return:
    """

    # These values are specific to CIFAR10
    img_rows = 32
    img_cols = 32
    nb_classes = 10

    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train /= 255
    X_test /= 255

    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    return X_train, Y_train, X_test, Y_test


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

    if not hasattr(backend, "tf"):
        raise RuntimeError("This tutorial requires keras to be configured"
                           " to use the TensorFlow backend.")

    # Image dimensions ordering should follow the Theano convention
    if keras.backend.image_dim_ordering() != 'tf':
        keras.backend.set_image_dim_ordering('tf')
        print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to "
              "'th', temporarily setting to 'tf'")

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
    #label_smooth = .1
    #Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)

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

    attack = FLAGS.attack
    attack_iterations = FLAGS.attack_iterations

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

    if adv != 0:
        print("Adversarial training")
        if adv == ATTACK_CARLINI_WAGNER_L2:
            from cleverhans.attacks import CarliniWagnerL2
            train_attack_params = {'binary_search_steps': 1,
                                   'max_iterations': FLAGS.nb_iter,
                                   'learning_rate': 0.1,
                                   #'batch_size': att_batch_size,
                                   'initial_const': 10,
                                   }
            train_attacker = CarliniWagnerL2(model, back='tf', sess=sess)

        if adv == ATTACK_MADRYETAL:
            from cleverhans.attacks import MadryEtAl
            train_attack_params = {'eps': eps, 'eps_iter': EPS_ITER,
                                   'nb_iter': FLAGS.nb_iter}
            #'nb_iter': FLAGS.nb_iter, 'ord': 2}
            train_attacker = MadryEtAl(model, sess=sess)

        elif adv == ATTACK_FGSM:
            from cleverhans.attacks import FastGradientMethod
            stddev = int(np.ceil((MAX_EPS * 255) // 2))
            train_attack_params = {'eps': tf.abs(tf.truncated_normal(
                shape=(batch_size, 1, 1, 1), mean=0, stddev=stddev))}
            train_attacker = FastGradientMethod(model, back='tf', sess=sess)

        # create the adversarial trainer
        train_attack_params.update({'clip_min': 0., 'clip_max': 1.})
        adv_x_train = train_attacker.generate(x, **train_attack_params)
        preds_adv_train = model.get_probs(adv_x_train)

    def evaluate():
        # Evaluate the accuracy of the CIFAR-10 model on clean test examples
        eval_params = {'batch_size': batch_size}
        acc = model_eval(
            sess, x, y, preds, X_test, Y_test, args=eval_params)
        report.clean_train_clean_eval = acc
        #assert X_test.shape[0] == test_end - test_start, X_test.shape
        print('Test accuracy on legitimate examples: %0.4f' % acc)

    if train_from_scratch:
        # optionally do additional adversarial training
        if adv != 0:
            adv_nb_epochs = nb_epochs - delay
            print("Normal training for %d epochs" % (delay))
            print("Adversarial training for %d epochs" % (adv_nb_epochs))
            train_params.update({'nb_epochs': delay})
            train_params.update({'reuse_global_step': tf.AUTO_REUSE})
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

        if adv != 0:
            print("Adversarial training for %d epochs" % (adv_nb_epochs))
            train_params.update({'nb_epochs': adv_nb_epochs})
            if test:
                model_train(sess, x, y, preds, X_train, Y_train, model=model,
                            predictions_adv=preds_adv_train, evaluate=evaluate, args=train_params,
                            save=save, rng=rng)
            else:
                model_train(sess, x, y, preds, X_train, Y_train, model=model,
                            predictions_adv=preds_adv_train, args=train_params,
                            save=save, rng=rng)
    else:
        #tf_model_load(sess, model_path)
        model_load(sess, model_path)
        print('Restored model from %s' % model_path)
        # evaluate()

    #batch_size = FLAGS.batch_size
    # Evaluate the accuracy of the CIFAR10 model on legitimate test examples
    eval_params = {'batch_size': batch_size}
    accuracy = model_eval(sess, x, y, preds, X_test, Y_test,
                          feed={phase: False}, args=eval_params)

    print('Test accuracy on legitimate test examples: {0}'.format(accuracy))

    # calculate the expected calibration error if this flag is provided:
    if FLAGS.ece:
        eval_params.update({'M': FLAGS.M})
        ece = model_ece(sess, x, y, preds, X_test, Y_test,
                        feed={phase: False}, args=eval_params)
        print('ECE on legitimate test examples: {0}'.format(ece))
        #conf = sess.run(preds, feed_dict={x: adv_img, phase: BN_TEST_PHASE})

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

    #ORD = 1
    ORD = np.inf

    if attack == ATTACK_CARLINI_WAGNER_L2:
        from cleverhans.attacks import CarliniWagnerL2
        attacker = CarliniWagnerL2(model, back='tf', sess=sess)
        attack_params = {'binary_search_steps': 1,
                         'max_iterations': attack_iterations,
                         'learning_rate': 0.1,
                         'batch_size': att_batch_size,
                         'initial_const': 10,
                         }
    elif attack == ATTACK_JSMA:
        from cleverhans.attacks import SaliencyMapMethod
        attacker = SaliencyMapMethod(model, back='tf', sess=sess)
        attack_params = {'batch_size': 10,
                         'gamma': 0.04,
                         'theta': 1.,
                         'symbolic_impl': True}
    elif attack == ATTACK_FGSM:
        from cleverhans.attacks import FastGradientMethod
        attacker = FastGradientMethod(model, back='tf', sess=sess)
        attack_params = {'eps': eps}
    elif attack == ATTACK_MADRYETAL:
        from cleverhans.attacks import MadryEtAl
        attacker = MadryEtAl(model, back='tf', sess=sess)
        attack_params = {'eps': eps, 'eps_iter': EPS_ITER,
                         'nb_iter': FLAGS.nb_iter,
                         'ord': ORD,
                         'rand_init': True
                         }
        #'nb_iter': FLAGS.nb_iter, 'ord': 2}
    elif attack == ATTACK_EAD:
        from cleverhans.attacks import ElasticNetMethod
        attacker = ElasticNetMethod(model, back='tf', sess=sess)
        attack_params = {'max_iterations': attack_iterations,
                         'binary_search_steps': 3,
                         'initial_const': 1,
                         'batch_size': att_batch_size
                         }
    elif attack == ATTACK_BIM:
        from cleverhans.attacks import BasicIterativeMethod
        attacker = BasicIterativeMethod(model, back='tf', sess=sess)
        attack_params = {'eps': eps, 'eps_iter': EPS_ITER,
                         'nb_iter': FLAGS.nb_iter,
                         'ord': ORD}                        
    else:
        print("Attack undefined")
        pass

    if FLAGS.sweep:
        attack_params.update({'clip_min': 0., 'clip_max': 1.})
        if ORD == np.inf:
            max_eps = 128
            epsilons = np.linspace(1, max_eps, max_eps)
        elif ORD == 2:
            max_eps = 2000
            epsilons = np.linspace(0, max_eps, max_eps // 20, endpoint=False)
        elif ORD == 1:
            max_eps = 100000
            epsilons = np.linspace(0, max_eps, max_eps // 1000, endpoint=False)
        else:
            pass
        for e in epsilons:
            attack_params.update({'eps': e / 255.})
            # X_test_adv = attacker.generate_np(
            #   adv_inputs, phase, **attack_params)
            X_test_adv = attacker.generate(x, **attack_params)
            preds_adv = model.get_probs(X_test_adv)

            # Evaluate the accuracy of the CIFAR10 model on adversarial
            # examples
            # adv_accuracy = model_eval(sess, x, y, preds, X_test_adv,
            # true_labels,
            adv_accuracy = model_eval(sess, x, y, preds_adv, adv_inputs, true_labels,
                                      feed={phase: False}, args=eval_params)
            '''
            diff = X_test_adv - adv_inputs
            norm = np.mean(np.sqrt(np.sum(np.square(diff), axis=(1, 2, 3))))
            print('%.2f, %.4f, %.4f' %
                  (e, adv_accuracy, norm))
            '''
            print('%.2f, %.4f' % (e, adv_accuracy))
            #print('%.4f' % adv_accuracy)
    else:
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
        print('Test accuracy on adversarial examples {0:.4f}'.format(
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

        vmin = np.min(diff)
        vmax = np.max(diff)

        import matplotlib.pyplot as plt
        for i in range(10):
            fig = plt.figure(figsize=(8, 4))
            ax1 = fig.add_subplot(131)
            ax1.get_xaxis().set_visible(False)
            ax1.get_yaxis().set_visible(False)
            ax1.imshow(adv_inputs[i].reshape(img_rows, img_cols, channels))

            ax2 = fig.add_subplot(132)
            ax2.get_xaxis().set_visible(False)
            ax2.get_yaxis().set_visible(False)
            ax2.imshow(diff[i].reshape(img_rows, img_cols, channels) * 10)

            ax3 = fig.add_subplot(133)
            ax3.get_xaxis().set_visible(False)
            ax3.get_yaxis().set_visible(False)
            ax3.imshow(X_test_adv[i].reshape(img_rows, img_cols, channels))
            plt.tight_layout(pad=0)
            plt.show()

    if FLAGS.save:
        npy_path = '/export/mlrg/gallowaa/Documents/challenges/madry/cifar10_challenge/test_attack.npy'
        print(X_test_adv.shape)
        if X_test_adv.shape != (10000, 32, 32, 3):
            print("Shape will not work with Madry challenge")
        X_test_adv *= 255.
        X_test_adv = np.clip(X_test_adv, 0, 255)  # ensure valid pixel range
        np.save(npy_path, X_test_adv)

    if FLAGS.plot:
        import matplotlib.pyplot as plt

        # Initialize our array for grid visualization
        if nb_filters == 64:
            rows_1 = cols_1 = 8
        elif nb_filters == 32:
            rows_1 = 4
            cols_1 = 8
        else:
            rows_1 = cols_1 = 4

        grid_shape_1 = (cols_1, rows_1, 8, 8, 3)
        grid_shape_2 = (cols_1, rows_1, 6, 6, nb_filters)

        nb_filters_out = nb_filters * 2

        grid_viz_data_first = np.zeros(grid_shape_1, dtype='f')
        #grid_viz_data_second = np.zeros(grid_shape_2, dtype='f')

        for i, layer_name in enumerate(model.layer_names):
            if 'Conv2D0' in layer_name:
                kernel_first = sess.run(model.layers[i].real_kernels)
                kernel_first = (kernel_first.reshape(8, 8, 3, nb_filters) - kernel_first.min()
                                ) / (kernel_first.max() - kernel_first.min())
            if 'Conv2D2' in layer_name:
                kernel_second = sess.run(model.layers[i].real_kernels)
                kernel_second = (kernel_second.reshape(6, 6, nb_filters, nb_filters_out) - kernel_second.min()
                                 ) / (kernel_second.max() - kernel_second.min())

        k = 0
        for i in range(cols_1):
            for j in range(rows_1):
                print(k)
                grid_viz_data_first[i, j] = kernel_first[:, :, :, k]
                #grid_viz_data_second[i, j] = kernel_second[:, :, :1, k]
                k += 1

        from cleverhans.utils import grid_visual
        fig1 = grid_visual(grid_viz_data_first)
        #fig2 = grid_visual(grid_viz_data_second)
        '''
        top=0.945, bottom=0.055, left=0.0, right=1.0, hspace=0.2, wspace=0.2
        '''
        if FLAGS.fooling:
            itr = 50
            print('Training knn...')

            from sklearn import neighbors
            knn = neighbors.KNeighborsClassifier(n_neighbors=2)
            knn.fit(X_train.reshape(X_train.shape[
                    0], img_rows * img_cols * channels), Y_train)
            print('Finished training knn')

            class_names = ['airplane', 'auto', 'bird', 'cat',
                           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

            np.random.seed(0)
            labels = np.zeros((1, nb_classes))
            conf_ar = np.zeros(nb_classes)
            path = './cifar10_cleverhans_gen'

            attack_params = {'eps': 1, 'eps_iter': 0.005,
                             'nb_iter': itr, 'clip_min': 0., 'clip_max': 1.}
            from cleverhans.attacks import BasicIterativeMethod
            attacker = BasicIterativeMethod(model, back='tf', sess=sess)

            #fig, axes = plt.subplots(1, 10, squeeze=True, figsize=(8, 1))

            # generate unrecognizable adversarial examples
            for i in range(nb_classes):

                print("Generating %s" % class_names[i])
                fig = plt.figure(figsize=(4, 4))
                ax1 = fig.add_subplot(111)
                ax1.get_xaxis().set_visible(False)
                ax1.get_yaxis().set_visible(False)
                adv_img = 0.5 + \
                    np.random.rand(1, img_rows, img_cols, channels) / 10

                labels[0, :] = 0
                labels[0, i] = 1

                attack_params.update({'y_target': labels})
                adv_img = attacker.generate_np(adv_img, **attack_params)

                conf = sess.run(preds, feed_dict={
                    x: adv_img, phase: BN_TEST_PHASE, logits_scalar: ATTACK_T})
                conf_ar[i] = np.max(conf)
                for k in range(nb_classes):
                    print('%d: %.4f' % (k, conf[0][k]))
                '''
                axes[i].imshow(adv_img.reshape(img_rows, img_cols, channels))
                axes[i].get_xaxis().set_visible(False)
                axes[i].get_yaxis().set_visible(False)
                '''
                ax1.imshow(adv_img.reshape(img_rows, img_cols, channels))
                plt.tight_layout(pad=0)
                plt.show()

                # find the nearest neighbor of this data point
                distance, idx = knn.kneighbors(adv_img.reshape(1, -1), 20)
                for j in range(idx[0].shape[0]):
                    if np.argmax(Y_train[idx[0][j]]) == np.argmax(labels):
                        print('%d correct neighbor at index %d, distance %f' %
                              (j, idx[0][j], distance[0][j]))
                        fig = plt.figure(figsize=(4, 4))
                        ax1 = fig.add_subplot(111)
                        ax1.get_xaxis().set_visible(False)
                        ax1.get_yaxis().set_visible(False)
                        ax1.imshow(X_train[idx[0][0]].reshape(
                            img_rows, img_cols, channels))
                        plt.tight_layout(pad=0)
                        plt.show()
                    else:
                        print('%d wrong neighbor at index %d, distance %f' %
                              (j, idx[0][j], distance[0][j]))
                if FLAGS.save:
                    fname = 'cifar10_%d_conf%.0f' % (i, conf[0][i] * 1000)
                    plt.savefig(os.path.join(path, fname + '.png'))
            plt.show()
            print('%.4f' % accuracy)
            print('%.4f' % adv_accuracy)
            print('%.4f' % percent_perturbed)
            print('%.4f' % conf_ar.mean())
    sess.close()


if __name__ == '__main__':

    par = argparse.ArgumentParser()

    # Generic flags
    par.add_argument('--gpu', help='id of GPU to use')
    par.add_argument('--model_path', help='Path to save or load model')
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
    par.add_argument('--attack', type=int, default=3,
                     help='Attack type, 0=CWL2, 1=JSMA, 2=FGSM, 3=PGD, 4=EAD')
    par.add_argument('--attack_iterations', type=int, default=100,
                     help='Number of iterations to run CW attack; 1000 is good')
    par.add_argument('--nb_samples', type=int,
                     default=1000, help='Nb of inputs to attack')
    par.add_argument(
        '--targeted', help='Run a targeted attack?', action="store_true")
    par.add_argument(
        '--fooling', help="run Nguyen et al. (2015) fooling images attack after training", action="store_true")
    par.add_argument(
        '--sweep', help='Run a targeted attack?', action="store_true")

    # Adversarial training flags
    par.add_argument(
        '--adv', help='Adversarial training type', type=int, default=0)
    par.add_argument('--delay', type=int,
                     default=5, help='Nb of epochs to delay adv training by')
    par.add_argument('--nb_iter', type=int,
                     default=7, help='Nb of iterations of PGD')

    # Confidence
    par.add_argument('--M', type=int, default=10)
    par.add_argument(
        '--ece', help='calculate the expected calibration error on clean test set', action="store_true")

    FLAGS = par.parse_args()

    if FLAGS.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    tf.app.run()
