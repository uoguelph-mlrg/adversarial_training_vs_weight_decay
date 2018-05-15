from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse
import logging
import scipy.misc
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorflow.python.platform import flags

from utils import (parse_model_settings,
                   model_load, data_cifar10)
from cleverhans.utils import set_log_level, AccuracyReport
from cleverhans.utils_tf import model_eval

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

    class_names = ['airplane', 'auto', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    np.random.seed(0)
    labels = np.zeros((1, nb_classes))
    conf_ar = np.zeros(nb_classes)
    path = './cifar10_cleverhans_gen'

    if FLAGS.viz:
        attack_params = {'eps': FLAGS.eps_iter, 'clip_min': 0., 'clip_max': 1.}
        from cleverhans.attacks import FastGradientMethod
        attacker = FastGradientMethod(model, back='tf', sess=sess)

        # generate unrecognizable adversarial examples
        for i in range(nb_classes):

            print("Generating %s" % class_names[i])

            fig = plt.figure(figsize=(4, 4))
            ax1 = fig.add_subplot(111)
            ax1.get_xaxis().set_visible(False)
            ax1.get_yaxis().set_visible(False)
            '''
            adv_img = 0.5 + \
                np.random.rand(1, img_rows, img_cols, channels) / 10
            '''                
            adv_img = np.random.rand(1, img_rows, img_cols, channels)
            '''
            adv_img = np.clip(np.random.normal(
                            loc=0.5, scale=0.25, size=(1, img_rows, img_cols, channels)), 0, 1)
            '''
            labels[0, :] = 0
            labels[0, i] = 1
            attack_params.update({'y_target': labels})

            for j in range(FLAGS.nb_iter):
                adv_img = attacker.generate_np(adv_img, **attack_params)
                #ax1.imshow(adv_img.reshape(img_rows, img_cols, channels))
                #plt.pause(0.05)
                scipy.misc.imsave(os.path.join(FLAGS.out_dir, 't%d_%d.png' % (i, j)), adv_img.reshape(32,32,3))
                
    else:
        # build a rectangle in axes coords
        left, width = .25, .5
        bottom, height = .25, .5
        right = left + width
        top = bottom + height
        fig, axes = plt.subplots(1, 10, squeeze=True, figsize=(8, 1.25))
        attack_params = {'eps': 1, 'eps_iter': FLAGS.eps_iter,
                         'nb_iter': FLAGS.nb_iter, 'clip_min': 0., 'clip_max': 1.}
        from cleverhans.attacks import BasicIterativeMethod
        attacker = BasicIterativeMethod(model, back='tf', sess=sess)

        # generate unrecognizable adversarial examples
        for i in range(nb_classes):

            print("Generating %s" % class_names[i])

            '''
            Draw some noise from a uniform or Gaussian distribution.
            these settings are fairly arbitrary, feel free to tune the knobs

            You may also want to try:
            adv_img = np.clip(np.random.normal(
                            loc=0.5, scale=0.25, size=(1, img_rows, img_cols, channels)), 0, 1)
            '''
            adv_img = 0.5 + \
                np.random.rand(1, img_rows, img_cols, channels) / 10

            labels[0, :] = 0
            labels[0, i] = 1

            attack_params.update({'y_target': labels})
            adv_img = attacker.generate_np(adv_img, **attack_params)
            axes[i].imshow(adv_img.reshape(img_rows, img_cols, channels))
            axes[i].get_xaxis().set_visible(False)
            axes[i].get_yaxis().set_visible(False)
            if FLAGS.annot:
                ax = axes[i]
                ax.text(0.5 * (left + right), 1.0, class_names[i], horizontalalignment='center',
                        verticalalignment='bottom', rotation=30, transform=ax.transAxes, size='larger')
                top = 0.6
            else:
                top = 1.0
            plt.tight_layout(pad=0)

        plt.subplots_adjust(left=0, bottom=0, right=1.0,
                            top=top, wspace=0.2, hspace=0.2)
        plt.show()
    sess.close()


if __name__ == '__main__':

    par = argparse.ArgumentParser()

    par.add_argument('--gpu', help='id of GPU to use')

    par.add_argument('--model_path', help='Path to model ckpt',
                     default='./ckpt/cnn-l2-model')

    par.add_argument('--data_dir', help='Path to CIFAR-10 data',
                     default='/scratch/gallowaa/cifar10/cifar10_data')

    par.add_argument('--out_dir', help='Path to CIFAR-10 data',
                     default='/scratch/gallowaa/cifar10/query/cnn-l2-model/fooling')

    par.add_argument('--nb_filters', type=int, default=32,
                     help='Number of filters in first layer')

    par.add_argument('--batch_size', type=int, default=100,
                     help='Size of evaluation batches')

    par.add_argument('--eps_iter', type=float, default=0.005, help='step size')

    par.add_argument('--nb_iter', type=int,
                     default=50, help='iterations of gradient ascent for crafting fooling images')
    par.add_argument(
        '--annot', help='annotate class labels on top of images', action="store_true")

    par.add_argument('--viz', help='viz each step', action="store_true")

    FLAGS = par.parse_args()

    if FLAGS.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    tf.app.run()
