from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse
import logging
import time
import numpy as np
import scipy.misc
import tensorflow as tf
from tqdm import tqdm
from tensorflow.python.platform import flags
from tensorflow.python.client import device_lib

from MadryLab.ilyas_utils import *
from utils import (parse_model_settings,
                   model_load, data_cifar10)
from cleverhans.utils import set_log_level, AccuracyReport
from cleverhans.utils_tf import model_eval

import matplotlib
import matplotlib.pyplot as plt

font = {'size': 18}
matplotlib.rc('font', **font)

FLAGS = flags.FLAGS

EPS_ITER = 2
MAX_BATCH_SIZE = 100
#OUT_DIR = '/scratch/ssd/gallowaa/cifar10/query/cnn-l2/1/'
#MOMENTUM = 0.9
# Things you can play around with:
BATCH_SIZE = 80
#SIGMA = 1e-5
EPSILON = 0.05
EPS_DECAY = 0.005
MIN_EPS_DECAY = 5e-5
LEARNING_RATE = 1e-4
#FLAGS.nb_samples = 400
MAX_LR = 1e-2
MIN_LR = 5e-5
# Things you probably don't want to change:
MAX_QUERIES = 4000000


def pseudorandom_target(index, total_indices, true_class):
    rng = np.random.RandomState(index)
    target = true_class
    while target == true_class:
        target = rng.randint(0, total_indices)
    return target


def pseudorandom_target_image(orig_index, total_indices):
    rng = np.random.RandomState(orig_index)
    target_img_index = orig_index
    while target_img_index == orig_index:
        target_img_index = rng.randint(0, total_indices)
    return target_img_index

'''
def get_image(index):
    data_path = os.path.join(IMAGENET_PATH, 'val')
    image_paths = sorted([os.path.join(data_path, i)
                          for i in os.listdir(data_path)])
    print(len(image_paths))
    # assert len(image_paths) == 50000
    labels_path = os.path.join(IMAGENET_PATH, 'val.txt')
    with open(labels_path) as labels_file:
        labels = [i.split(' ') for i in labels_file.read().strip().split('\n')]
        labels = {os.path.basename(i[0]): int(i[1]) for i in labels}

    def get(index):
        path = image_paths[index]
        x = load_image(path)
        y = labels[os.path.basename(path)]
        return x, y
    return get(index)
'''


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


if __name__ == '__main__':

    par = argparse.ArgumentParser()

    par.add_argument('--gpu', help='id of GPU to use')

    par.add_argument('--model_path', help='Path to model ckpt',
                     default='./ckpt/cnn-l2-model')

    par.add_argument('--out_dir', help='save path')

    par.add_argument('--data_dir', help='Path to CIFAR-10 data',
                     default='/scratch/gallowaa/cifar10/cifar10_data')

    par.add_argument('--nb_filters', type=int, default=32,
                     help='Number of filters in first layer')

    par.add_argument('--img', type=int, default=0,
                     help='Number of filters in first layer')

    par.add_argument('--batch_size', type=int, default=100,
                     help='Size of evaluation batches')

    par.add_argument("--attack", help='which attack to evaluate with', type=str,
                     default='fgsm', choices=['fgsm', 'pgd', 'cwl2', 'jsma'])

    par.add_argument('--eps', type=float, default=8, help='epsilon')

    par.add_argument('--momentum', type=float, default=0, help='momentum')

    par.add_argument('--sigma', type=float, default=1e-3, help='sigma')

    par.add_argument('--nb_samples', type=int,
                     default=800, help='Nb of samples_per_draw')

    par.add_argument(
        '--targeted', help='Run a targeted attack?', action="store_true")

    par.add_argument(
        '--antithetic', help='antithetic sampling', action="store_true")

    par.add_argument('--nb_iter', type=int,
                     default=20, help='iterations of PGD')

    FLAGS = par.parse_args()

    if FLAGS.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    # if FLAGS.out_dir:
    #        OUT_DIR = FLAGS.out_dir

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

    if FLAGS.attack == 'fgsm':
        from cleverhans.attacks import FastGradientMethod
        attacker = FastGradientMethod(model, back='tf', sess=sess)
        attack_params = {'eps': FLAGS.eps / 255., yname: adv_ys}
    elif FLAGS.attack == 'pgd':
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

    # Query-efficient gradient estimator here
    if FLAGS.out_dir is not None:
        out_dir = build_query_save_path(
            FLAGS.out_dir, model_path, FLAGS.img, FLAGS.nb_samples, FLAGS.sigma, FLAGS.momentum, FLAGS.antithetic)
        summary_writer = tf.summary.FileWriter(out_dir, sess.graph)

    k = 10
    print('Starting partial-information attack with only top-' + str(k))
    num_indices = X_test.shape[0]
    IMG_INDEX = FLAGS.img
    target_image_index = pseudorandom_target_image(IMG_INDEX, num_indices)

    # x, y = get_image(IMG_INDEX)
    orig_class = np.argmax(Y_test[IMG_INDEX]).astype('int32')
    initial_img = X_test[IMG_INDEX]

    target_img = None
    # target_img, _ = get_image(target_image_index)
    target_img = X_test[target_image_index]

    target_class = orig_class
    print('Set target class to be original img class %d for partial-info attack' % target_class)
    batch_size = min(BATCH_SIZE, FLAGS.nb_samples)
    assert FLAGS.nb_samples % BATCH_SIZE == 0
    one_hot_vec = one_hot(target_class, nb_classes)

    gpus = [get_available_gpus()[0]]
    labels = np.repeat(np.expand_dims(one_hot_vec, axis=0),
                       repeats=batch_size, axis=0)

    grad_estimates = []
    final_losses = []
    if FLAGS.antithetic:
        noise_pos = tf.random_normal(
            (batch_size // 2,) + initial_img.shape)
        noise = tf.concat([noise_pos, -noise_pos], axis=0)
    else:
        noise = tf.random_normal(
            (batch_size,) + initial_img.shape)
    eval_points = x + FLAGS.sigma * noise
    logits = model.get_logits(eval_points)
    losses = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=labels)
    # inds is batch_size x k
    # returns (# true) x 3
    vals, inds = tf.nn.top_k(logits, k=k)

    good_inds = tf.where(tf.equal(inds, tf.constant(target_class)))
    good_images = good_inds[:, 0]  # inds of img in batch that worked
    losses = tf.gather(losses, good_images)
    noise = tf.gather(noise, good_images)

    losses_tiled = tf.tile(tf.reshape(
        losses, (-1, 1, 1, 1)), (1,) + initial_img.shape)
    grad_estimates.append(tf.reduce_mean(losses_tiled * noise,
                                         axis=0) / FLAGS.sigma)
    final_losses.append(losses)

    grad_estimate = tf.reduce_mean(grad_estimates, axis=0)
    '''
    g_u, g_v = tf.nn.moments(
        tf.abs(grad_estimates), axes=[0], keep_dims=False)
    tf.summary.scalar("stats/attk_grad_u", tf.reduce_mean(g_u))
    tf.summary.scalar("stats/att_grad_v", tf.reduce_mean(g_v))
    '''
    grad_u, grad_var = tf.nn.moments(
        tf.abs(grad_estimate), axes=[0], keep_dims=False)
    tf.summary.scalar("stats/grad_u", tf.reduce_mean(grad_u))
    tf.summary.scalar("stats/grad_v", tf.reduce_mean(grad_var))

    final_losses = tf.concat(final_losses, axis=0)
    tf.summary.scalar("final_losses", tf.reduce_mean(final_losses))
    merge_op = tf.summary.merge_all()

    eval_logits = model.get_logits(x)
    eval_preds = tf.argmax(eval_logits, 1)
    eval_adv = tf.reduce_sum(tf.to_float(
        tf.equal(eval_preds, target_class)))

    samples_per_draw = FLAGS.nb_samples

    def get_grad(pt, summary_writer, merge_op, i, should_calc_truth=False):
        num_batches = samples_per_draw // batch_size
        losses = []
        grads = []
        feed_dict = {x: pt}
        for _ in range(num_batches):
            loss, dl_dx_, summ = sess.run(
                [final_losses, grad_estimate, merge_op], feed_dict)
            losses.append(np.mean(loss))
            grads.append(dl_dx_)
        summary_writer.add_summary(summ, i)
        summary_writer.flush()
        return np.array(losses).mean(), np.mean(np.array(grads), axis=0)

    with tf.device('/cpu:0'):
        render_feed = tf.placeholder(tf.float32, initial_img.shape)
        render_exp = tf.expand_dims(render_feed, axis=0)
        render_logits = model.get_logits(render_exp)

    def render_frame(image, save_index):
        # actually draw the figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 8))
        # image
        ax1.imshow(image)
        fig.sca(ax1)
        plt.xticks([])
        plt.yticks([])
        # classifications
        probs = softmax(sess.run(render_logits, {render_feed: image})[0])
        topk = probs.argsort()[-5:][::-1]
        topprobs = probs[topk]
        # print('topk')
        # print(topk)
        barlist = ax2.bar(range(5), topprobs)
        for i, v in enumerate(topk):
            if v == orig_class:
                barlist[i].set_color('g')
            if v == target_class:
                barlist[i].set_color('r')
        plt.sca(ax2)
        plt.ylim([0, 1.1])
        plt.xticks(range(5), [label_to_name(i)
                              for i in topk], rotation='vertical')
        fig.subplots_adjust(bottom=0.2)

        if out_dir is not None:
            path = os.path.join(out_dir, 'frame%06d.png' % save_index)
            if os.path.exists(path):
                os.remove(path)
        plt.savefig(path)
        plt.close()

    adv = initial_img.copy().reshape(1, 32, 32, 3)
    #assert out_dir[-1] == '/'

    if FLAGS.out_dir is not None:
        log_file = open(os.path.join(out_dir, 'log.txt'), 'w+')
    g = 0
    num_queries = 0

    last_ls = []
    current_lr = LEARNING_RATE

    max_iters = int(np.ceil(MAX_QUERIES // FLAGS.nb_samples))
    real_eps = 0.5

    lrs = []
    max_lr = MAX_LR
    epsilon_decay = EPS_DECAY
    last_good_adv = adv
    for i in range(max_iters):
        start = time.time()
        if FLAGS.out_dir is not None:
            render_frame(adv.reshape(32, 32, 3), i)

        # see if we should stop
        padv = sess.run(eval_adv, feed_dict={x: adv})
        if (padv == 1) and (real_eps <= EPSILON):
            print('partial info early stopping at iter %d' % i)
            break

        assert target_img is not None
        lower = np.clip(target_img - real_eps, 0., 1.)
        upper = np.clip(target_img + real_eps, 0., 1.)
        prev_g = g
        l, g = get_grad(adv, summary_writer, merge_op, i)

        if l < 0.2:
            real_eps = max(EPSILON, real_eps - epsilon_decay)
            max_lr = MAX_LR
            last_good_adv = adv
            epsilon_decay = EPS_DECAY
            if real_eps <= EPSILON:
                samples_per_draw = 5000
            last_ls = []

        # simple momentum
        g = FLAGS.momentum * prev_g + (1.0 - FLAGS.momentum) * g

        last_ls.append(l)
        last_ls = last_ls[-5:]
        if last_ls[-1] > last_ls[0] and len(last_ls) == 5:
            if max_lr > MIN_LR:
                print("ANNEALING MAX LR")
                max_lr = max(max_lr / 2.0, MIN_LR)
            else:
                print("ANNEALING EPS DECAY")
                adv = last_good_adv  # start over with a smaller eps
                l, g = get_grad(adv, summary_writer, merge_op, i)
                assert (l < 1)
                epsilon_decay = max(epsilon_decay / 2, MIN_EPS_DECAY)
            last_ls = []

        # backtracking line search for optimal lr
        current_lr = max_lr
        while current_lr > MIN_LR:
            proposed_adv = adv - current_lr * np.sign(g)
            proposed_adv = np.clip(proposed_adv, lower, upper)
            num_queries += 1
            eval_logits_ = sess.run(eval_logits, {x: proposed_adv})[0]
            if target_class in eval_logits_.argsort()[-k:][::-1]:
                lrs.append(current_lr)
                adv = proposed_adv
                break
            else:
                current_lr = current_lr / 2
                print('backtracking, lr = %.2E' % current_lr)

        num_queries += FLAGS.nb_samples

        log_text = 'Step %05d: loss %.4f eps %.4f eps-decay %.4E lr %.2E (time %.4f)' % (i, l,
                                                                                         real_eps, epsilon_decay, current_lr, time.time() - start)
        #log_file.write(log_text + '\n')
        print(log_text)

        #np.save(os.path.join(FLAGS.out_dir, '%s.npy' % (i + 1)), adv)
        if FLAGS.out_dir is not None:
            scipy.misc.imsave(os.path.join(out_dir, '%s.png' % (i + 1)), adv.reshape(32,32,3))
        #summary_writer.add_summary(merged_summ, step)
        # summary_writer.flush()
