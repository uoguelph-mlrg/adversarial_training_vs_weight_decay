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
import argparse
import logging
import time
import scipy.misc
from keras.utils import np_utils
from cleverhans.utils import set_log_level
from cleverhans.utils_tf import model_eval
import cifar10_input

import matplotlib.pyplot as plt
from ilyas_utils import *

EPS_ITER = 2
MAX_BATCH_SIZE = 100
#OUT_DIR = '/scratch/ssd/gallowaa/cifar10/query/natural/'
OUT_DIR = '/scratch/ssd/gallowaa/cifar10/query/madry/'
#MOMENTUM = 0.0
# Things you can play around with:
BATCH_SIZE = 40
#SIGMA = 1e-3
EPSILON = 0.05 * 255.
EPS_DECAY = 0.005 * 255.
MIN_EPS_DECAY = 5e-5 * 255
LEARNING_RATE = 1e-4
#SAMPLES_PER_DRAW = 800
MAX_LR = 1e-2
MIN_LR = 5e-5
# Things you probably don't want to change:
MAX_QUERIES = 4000000

if __name__ == '__main__':
    import time
    import json
    import sys
    import math
    import argparse

    par = argparse.ArgumentParser()

    par.add_argument('--gpu', help='id of GPU to use')

    par.add_argument('--model_path', help='Path to model ckpt',
                     default='/export/mlrg/gallowaa/Documents/challenges/madry/cifar10_challenge_upstream/models/naturally_trained')

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

    with open('config.json') as config_file:
        config = json.load(config_file)

    #model_file = tf.train.latest_checkpoint(config['model_dir'])
    model_file = tf.train.latest_checkpoint(FLAGS.model_path)

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
        '''
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
        '''
        # Query-efficient gradient estimator here

        out_dir = build_query_save_path(
            OUT_DIR, FLAGS.model_path, FLAGS.img, FLAGS.nb_samples, FLAGS.sigma, FLAGS.momentum, FLAGS.antithetic)
        summary_writer = tf.summary.FileWriter(out_dir, sess.graph)

        # Query-efficient gradient estimator here
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

        #gpus = [get_available_gpus()[0]]
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
        final_losses = tf.concat(final_losses, axis=0)

        grad_u, grad_var = tf.nn.moments(
            tf.abs(grad_estimate), axes=[0], keep_dims=False)
        tf.summary.scalar("stats/grad_u", tf.reduce_mean(grad_u))
        tf.summary.scalar("stats/grad_v", tf.reduce_mean(grad_var))

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
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
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

            path = os.path.join(out_dir, 'frame%06d.png' % save_index)
            if os.path.exists(path):
                os.remove(path)
            plt.savefig(path)
            plt.close()

        adv = initial_img.copy().reshape(1, 32, 32, 3)
        #assert out_dir[-1] == '/'

        log_file = open(os.path.join(out_dir, 'log.txt'), 'w+')
        g = 0
        num_queries = 0

        last_ls = []
        current_lr = LEARNING_RATE

        max_iters = int(np.ceil(MAX_QUERIES // FLAGS.nb_samples))
        real_eps = 0.5 * 255.

        lrs = []
        max_lr = MAX_LR
        epsilon_decay = EPS_DECAY
        last_good_adv = adv
        for i in range(max_iters):
            start = time.time()
            # if i % 10 == 0:
            render_frame(adv.reshape(32, 32, 3), i)

            # see if we should stop
            padv = sess.run(eval_adv, feed_dict={x: adv})
            if (padv == 1) and (real_eps <= EPSILON):
                print('partial info early stopping at iter %d' % i)
                break

            assert target_img is not None
            lower = np.clip(target_img - real_eps, 0., 255.)
            upper = np.clip(target_img + real_eps, 0., 255.)
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
                    l, g = get_grad(adv)
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

            #np.save(os.path.join(out_dir, '%s.npy' % (i + 1)), adv)
            scipy.misc.imsave(os.path.join(out_dir, '%s.png' %
                                           (i + 1)), adv.reshape(32, 32, 3))
