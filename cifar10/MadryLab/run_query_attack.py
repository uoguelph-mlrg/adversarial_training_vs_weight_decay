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
OUT_DIR = '/scratch/ssd/gallowaa/cifar10/madry/natural/'
MOMENTUM = 0.0
# Things you can play around with:
BATCH_SIZE = 40
SIGMA = 1e-3
EPSILON = 0.05 * 255.
EPS_DECAY = 0.005 * 255.
MIN_EPS_DECAY = 5e-5 * 255
LEARNING_RATE = 1e-4
SAMPLES_PER_DRAW = 1000
MAX_LR = 1e-2
MIN_LR = 5e-5
# Things you probably don't want to change:
MAX_QUERIES = 4000000

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
        k = 10
        print('Starting partial-information attack with only top-' + str(k))
        num_indices = X_test.shape[0]
        IMG_INDEX = 2
        target_image_index = pseudorandom_target_image(IMG_INDEX, num_indices)

        # x, y = get_image(IMG_INDEX)
        orig_class = np.argmax(Y_test[IMG_INDEX]).astype('int32')
        initial_img = X_test[IMG_INDEX]

        target_img = None
        # target_img, _ = get_image(target_image_index)
        target_img = X_test[target_image_index]

        target_class = orig_class
        print('Set target class to be original img class %d for partial-info attack' % target_class)
        batch_size = min(BATCH_SIZE, SAMPLES_PER_DRAW)
        assert SAMPLES_PER_DRAW % BATCH_SIZE == 0
        one_hot_vec = one_hot(target_class, nb_classes)

        #gpus = [get_available_gpus()[0]]
        labels = np.repeat(np.expand_dims(one_hot_vec, axis=0),
                           repeats=batch_size, axis=0)
        grad_estimates = []
        final_losses = []
        noise_pos = tf.random_normal(
            (batch_size // 2,) + initial_img.shape)
        noise = tf.concat([noise_pos, -noise_pos], axis=0)
        eval_points = x + SIGMA * noise
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
                                             axis=0) / SIGMA)
        final_losses.append(losses)
        grad_estimate = tf.reduce_mean(grad_estimates, axis=0)
        final_losses = tf.concat(final_losses, axis=0)

        eval_logits = model.get_logits(x)
        eval_preds = tf.argmax(eval_logits, 1)
        eval_adv = tf.reduce_sum(tf.to_float(
            tf.equal(eval_preds, target_class)))

        samples_per_draw = SAMPLES_PER_DRAW

        def get_grad(pt, should_calc_truth=False):
            num_batches = samples_per_draw // batch_size
            losses = []
            grads = []
            feed_dict = {x: pt}
            for _ in range(num_batches):
                loss, dl_dx_ = sess.run([final_losses, grad_estimate], feed_dict)
                losses.append(np.mean(loss))
                grads.append(dl_dx_)
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
            #print('topk')
            #print(topk)
            barlist = ax2.bar(range(5), topprobs)
            for i, v in enumerate(topk):
                if v == orig_class:
                    barlist[i].set_color('g')
                if v == target_class:
                    barlist[i].set_color('r')
            plt.sca(ax2)
            plt.ylim([0, 1.1])
            plt.xticks(range(5), [label_to_name(i) for i in topk], rotation='vertical')
            fig.subplots_adjust(bottom=0.2)

            path = os.path.join(OUT_DIR, 'frame%06d.png' % save_index)
            if os.path.exists(path):
                os.remove(path)
            plt.savefig(path)
            plt.close()

        adv = initial_img.copy().reshape(1, 32, 32, 3)
        assert OUT_DIR[-1] == '/'

        log_file = open(os.path.join(OUT_DIR, 'log.txt'), 'w+')
        g = 0
        num_queries = 0

        last_ls = []
        current_lr = LEARNING_RATE

        max_iters = int(np.ceil(MAX_QUERIES // SAMPLES_PER_DRAW))
        real_eps = 0.5 * 255.

        lrs = []
        max_lr = MAX_LR
        epsilon_decay = EPS_DECAY
        last_good_adv = adv
        for i in range(max_iters):
            start = time.time()
            if i % 10 == 0:
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
            l, g = get_grad(adv)

            if l < 0.2:
                real_eps = max(EPSILON, real_eps - epsilon_decay)
                max_lr = MAX_LR
                last_good_adv = adv
                epsilon_decay = EPS_DECAY
                if real_eps <= EPSILON:
                    samples_per_draw = 5000
                last_ls = []

            # simple momentum
            g = MOMENTUM * prev_g + (1.0 - MOMENTUM) * g

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

            num_queries += SAMPLES_PER_DRAW

            log_text = 'Step %05d: loss %.4f eps %.4f eps-decay %.4E lr %.2E (time %.4f)' % (i, l,
                                                                                             real_eps, epsilon_decay, current_lr, time.time() - start)
            #log_file.write(log_text + '\n')
            print(log_text)

            #np.save(os.path.join(OUT_DIR, '%s.npy' % (i + 1)), adv)
            scipy.misc.imsave(os.path.join(OUT_DIR, '%s.png' % (i + 1)), adv.reshape(32,32,3))
