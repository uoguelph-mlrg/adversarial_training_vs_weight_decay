from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import os.path
import argparse
import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

from utils import (fw, restore_model,
                   disp_data, disp_index_data)
LABEL_3 = 1
LABEL_7 = 0
FULL_SCALE = 255
EPS_MAX = 0.25

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # data src, ckpt, and logging
    parser.add_argument(
        '--restore', help='path to checkpoint to be loaded')
    parser.add_argument(
        '--data_dir', help='directory for storing input data', default='./')
    parser.add_argument(
        '--log_dir', help='root path for logging events and checkpointing')
    parser.add_argument(
        '--name', help='a unique name for the ckpt, to be used with log_dir')

    # expert initialization
    parser.add_argument(
        '--expert', help="initialize weights to avg 7 minus avg 3 from train set", action="store_true")

    # attack settings
    parser.add_argument(
        '--eps', help='FGSM eps', type=float, default=0.25)
    parser.add_argument("--attack", help='fast gradient method (FGM) variant', type=str,
                        default='fgsm', choices=['fsgm', 'fast_grad'])
    parser.add_argument(
        '--n_px', help='set most important n_px to 0', type=int, default=0)
    parser.add_argument(
        '--sweep', help="attack over a range of epsilon, displaying acc and mse as csv", action="store_true")

    # training hyper-parameters
    parser.add_argument(
        '--train', help="do training for max_steps", action="store_true")
    parser.add_argument(
        '--max_steps', help='number of steps to train for', type=int, default=10000)
    parser.add_argument(
        '--adv', help="perform one step fgsm adversarial training", action="store_true")
    parser.add_argument(
        '--l1', help='L1 weight decay regularization constant', type=float, default=0)
    parser.add_argument(
        '--l2', help='L2 weight decay regularization constant', type=float, default=0)
    parser.add_argument(
        '--learning_rate', help='learning rate', type=float, default=1e-5)
    parser.add_argument(
        '--std', help='stddev of truncated_normal used to init weights', type=float, default=1e-1)
    parser.add_argument(
        '--batch_size', help='examples per mini-batch', type=int, default=128)
    parser.add_argument(
        '--eval_every_n', help='validate model every n steps', type=int, default=100)
    parser.add_argument(
        '--bits', help='number of bits used to represent each weight', type=int, default=32)

    # visualizations
    parser.add_argument(
        '--show_perturbation', help="display perturbation used to create adversarial examples", action="store_true")
    parser.add_argument(
        '--show_weights', help="display weights after training or from restored model", action="store_true")
    parser.add_argument(
        '--show_examples', help="display some adversarial examples with perturbation given by 'eps' flag, do not pass 'sweep' flag", action="store_true")
    parser.add_argument(
        '--gen_examples', help="generate examples from the model", action="store_true")

    # misc options
    parser.add_argument(
        '--gpu', help='physical id of GPUs to use')

    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(0)

    # import data
    dtype = tf.float32
    mnist = input_data.read_data_sets(
        args.data_dir, dtype=dtype, one_hot=True)

    train_lab = np.argmax(mnist.train.labels, 1)
    test_lab = np.argmax(mnist.test.labels, 1)

    # Create 3 vs 7 training set
    train_mask = ((train_lab == 3) | (train_lab == 7))
    train_x = mnist.train.images[train_mask]
    train_y = mnist.train.labels[train_mask]
    train_labels = np.zeros((train_y.shape[0], 1))
    train_labels[np.argmax(train_y, 1) == 7] = LABEL_7
    train_labels[np.argmax(train_y, 1) == 3] = LABEL_3

    # Create 3 vs 7 test set
    test_mask = ((test_lab == 3) | (test_lab == 7))
    test_x = mnist.test.images[test_mask]
    test_y = mnist.test.labels[test_mask]
    test_labels = np.zeros((test_y.shape[0], 1))
    test_labels[np.argmax(test_y, 1) == 7] = LABEL_7
    test_labels[np.argmax(test_y, 1) == 3] = LABEL_3

    sevens = train_x[train_labels[:, 0] == LABEL_7]
    threes = train_x[train_labels[:, 0] == LABEL_3]

    if args.expert:
        avg_three = np.mean(threes, axis=0)
        avg_seven = np.mean(sevens, axis=0)
        diff = avg_three - avg_seven

    print('Training with %d images' % train_x.shape[0])
    print('Testing with %d images' % test_x.shape[0])

    with tf.Graph().as_default():

        global_step = tf.Variable(0, trainable=False)
        x = tf.placeholder(dtype, shape=(None, 784))
        y_ = tf.placeholder(dtype, shape=(None, 1))
        eps_ = tf.placeholder_with_default(
            args.eps, shape=(), name="epsilon")

        w = tf.Variable(tf.truncated_normal([784, 1], stddev=args.std))
        wb = fw(w, args.bits)

        # the logistic regression model
        y = tf.matmul(x, wb)
        loss_proxy = tf.reduce_mean(tf.nn.softplus(y_ * y))
        preds = tf.nn.sigmoid(y)
        avg_conf = tf.reduce_mean(2 * tf.abs(preds - 0.5))
        pred_thresh = tf.round(preds)

        loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y))

        adv_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=pred_thresh, logits=y))

        w_l1_norm = tf.reduce_mean(tf.abs(w))
        logits_l1_norm = tf.reduce_mean(tf.abs(y))

        # add various weight decay penalties
        l2_penalty = args.l2 * tf.nn.l2_loss(w)
        l1_penalty = args.l1 * w_l1_norm
        loss += l1_penalty + l2_penalty

        grad_loss_wrt_x = tf.gradients(loss, x)[0]

        if args.attack == 'fgsm':
            eta = eps_ * tf.sign(grad_loss_wrt_x)
        elif args.attack == 'fast_grad':
            red_ind = list(range(1, len(x.get_shape())))
            square = tf.reduce_sum(tf.square(grad_loss_wrt_x),
                                   reduction_indices=red_ind,
                                   keep_dims=True)
            normalized_grad = grad_loss_wrt_x / tf.sqrt(square)
            '''
            we multiply normalized_grad by 100 before clipping so that
            it more closely approximates the strength of the sign method.
            '''
            eta = tf.clip_by_value(100 * normalized_grad, -eps_, eps_)
        else:
            print("Attack %s not supported" % args.attack)

        # apply the perturbation
        x_adv = tf.clip_by_value(x + eta, 0, 1)
        x_opt = tf.clip_by_value(x - eta, 0, 1)

        train_op = tf.train.AdamOptimizer(args.learning_rate).minimize(loss)

        correct_prediction = tf.equal(pred_thresh, y_)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        saver = tf.train.Saver(tf.global_variables())

        init_step = 0
        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            if args.expert:
                sess.run(tf.assign(w, diff.reshape(784, 1)))
            elif args.restore:
                restore_model(sess, saver, args.restore)

            if args.train:
                batches_per_epoch = np.floor(
                    train_x.shape[0] / args.batch_size) - 1
                for step in range(args.max_steps):
                    s = int(step * args.batch_size % batches_per_epoch)
                    e = int(s + args.batch_size)
                    start_time = time.time()
                    # adversarial training
                    if args.adv:
                        batch_adv = sess.run(x_adv, feed_dict={
                            x: train_x[s:e], y_: train_labels[s:e]})
                        __, logits_l1, w_l1, loss_val, loss_sp = sess.run([train_op, logits_l1_norm, w_l1_norm, loss_proxy, loss], feed_dict={
                            x: batch_adv, y_: train_labels[s:e]})
                    # normal training
                    else:
                        __, logits_l1, w_l1, loss_val, loss_sp, p = sess.run([train_op, logits_l1_norm, w_l1_norm, loss, loss_proxy, eta], feed_dict={
                            x: train_x[s:e], y_: train_labels[s:e]})
                    duration = time.time() - start_time

                    if step % args.eval_every_n == 0:
                        test_acc = sess.run(accuracy, feed_dict={
                                            x: test_x, y_: test_labels})
                        print("step %d, logits_l1=%.3f, w_l1=%.3f, loss=%.3f, loss_sp=%.3f, test acc. %.4f (%.1f ex/s)" %
                              (step, logits_l1, w_l1, loss_val, loss_sp, test_acc, float(args.batch_size / duration)))

                if args.log_dir:
                    # save model checkpoint to log_dir if provided
                    checkpoint_path = os.path.join(
                        args.log_dir, args.name + '_model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

            # Test model
            cln_acc = sess.run(accuracy, feed_dict={
                               x: test_x, y_: test_labels})
            print("Clean test accuracy %.4f" % (cln_acc))

            if args.show_weights:
                w_np = sess.run(wb)
                disp_data(w_np)

            # The ``L_0'' attack
            if args.n_px:
                w_np = sess.run(wb)
                w_sorted = np.argsort(w_np, axis=0)
                for i in range(args.n_px):
                    test_x[:, w_sorted[784 - i:].reshape(i)] = 1
                    test_x[:, w_sorted[:i].reshape(i)] = 1
                    px_acc, avg_c = sess.run([accuracy, avg_conf], feed_dict={
                        x: test_x, y_: test_labels})
                    print("%.4f, %.4f" % (px_acc, avg_c))

                # viz the examples
                if args.show_examples:
                    for i in range(10):
                        conf, p = sess.run([preds, pred_thresh], feed_dict={
                            x: test_x[i, :].reshape(1, 784)})
                        print('%d: %d: %.4f' %
                              (test_labels[i, 0], p[0, 0], conf[0, 0]))
                        disp_data(test_x[i, :])

            if args.show_perturbation:
                eta_np = sess.run(eta, feed_dict={x: test_x, y_: test_labels})
                disp_index_data(eta_np, 0)

            if args.sweep:
                epsilons = np.linspace(0.05, 0.5, 50)
                print("epsilon, accuracy, mse")
                for i, eps in enumerate(epsilons):
                    adv_np = sess.run(x_adv, feed_dict={
                        x: test_x, y_: test_labels, eps_: eps})
                    adv_acc = sess.run(accuracy, feed_dict={
                        x: adv_np, y_: test_labels})
                    mse = np.mean(np.sum((adv_np - test_x)**2, axis=(1))**.5)
                    print("%.2f, %.4f, %.4f" % (eps, adv_acc, mse))
            else:
                adv_np, conf = sess.run([x_adv, preds], feed_dict={
                                        x: test_x, y_: test_labels})
                adv_acc = sess.run(accuracy, feed_dict={
                                   x: adv_np, y_: test_labels})
                print("FGSM test accuracy for eps %.2f = %.4f" %
                      (args.eps, adv_acc))
                mse = np.mean(np.sum((adv_np - test_x)**2, axis=(1))**.5)
                print("%.4f" % cln_acc)
                print("%.4f" % adv_acc)
                print('%.4f' % mse)

            # generate unrecognizable adversarial examples
            if args.gen_examples:
                labels = np.zeros((1, 1))
                for i in range(2):
                    adv_img = 0.5 * (np.random.rand(1, 784) / 10)
                    if i == 1:
                        labels[0, 0] = 1
                    adv_img = sess.run(x_opt, feed_dict={
                                       x: adv_img, y_: labels})
                    conf = sess.run(preds, feed_dict={x: adv_img})
                    print('%d: %.4f' % (i, conf[0][0]))
                    disp_data(adv_img)

        if args.show_examples and not args.sweep:
            '''
            Loop through 10 candidate adversarial examples, one at a time. 
            Click the close button to go to the next example
            '''
            for i in range(10):
                print("Label %d, prediction %.4f" % (test_labels[i], conf[i]))
                disp_index_data(adv_np, i)
