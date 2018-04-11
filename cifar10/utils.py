import os
import sys
import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from keras.utils import np_utils


def model_load(sess, ckpt_path):
    model_tokens = ckpt_path.split('/')
    with sess.as_default():
        saver = tf.train.Saver()
        if 'model' in model_tokens[-1]:
            saver.restore(sess, ckpt_path)
        else:
            saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))


def parse_model_settings(model_path):

    tokens = model_path.split('/')
    precision_list = ['bin', 'binsc', 'fp']
    precision = ''
    start_index = 0
    adv = False

    for p in precision_list:
        if p in tokens:
            start_index = tokens.index(p)
            precision = p
    try:
        binary = True if precision == 'bin' or precision == 'binsc' else False
        scale = True if precision == 'binsc' else False
        nb_filters = int(tokens[start_index + 1].split('_')[1])
        batch_size = int(tokens[start_index + 2].split('_')[1])
        learning_rate = float(tokens[start_index + 3].split('_')[1])
        nb_epochs = int(tokens[start_index + 4].split('_')[1])

        adv_index = start_index + 5
        if adv_index < len(tokens):
            adv = True if 'adv' in tokens[adv_index] else False

        print("Got %s model" % precision)
        print("Got %d filters" % nb_filters)
        print("Got batch_size %d" % batch_size)
        print("Got batch_size %f" % learning_rate)
        print("Got %d epochs" % nb_epochs)
    except:
        print("Could not parse tokens!")
        sys.exit(1)

    return binary, scale, nb_filters, batch_size, learning_rate, nb_epochs, adv


def build_model_save_path(root_path, binary, batch_size, nb_filters, lr, epochs, adv, delay, scale):

    precision = 'bin' if binary else 'fp'
    precision += 'sc/' if scale else '/'
    model_path = os.path.join(root_path, precision)
    model_path += 'k_' + str(nb_filters) + '/'
    model_path += 'bs_' + str(batch_size) + '/'
    model_path += 'lr_' + str(lr) + '/'
    model_path += 'ep_' + str(epochs)

    if adv != 0:
        model_path += '/adv_%d' % delay

    # optionally create this dir if it does not already exist,
    # otherwise, increment
    model_path = create_dir_if_not_exists(model_path)

    return model_path


def build_linear_model_save_path(root_path, batch_size, lr, epochs, adv, delay):

    model_path = os.path.join(root_path, 'bs_' + str(batch_size))
    model_path = os.path.join(model_path, 'lr_' + str(lr))
    model_path = os.path.join(model_path, 'ep_' + str(epochs))

    if adv:
        model_path = os.path.join(model_path, 'adv_%d' % delay)

    # optionally create this dir if it does not already exist,
    # otherwise, increment
    model_path = create_dir_if_not_exists(model_path)

    return model_path


def create_dir_if_not_exists(path):
    if not os.path.exists(path):
        path += '/1'
        os.makedirs(path)
    else:
        digits = []
        sub_dirs = next(os.walk(path))[1]
        [digits.append(s) for s in sub_dirs if s.isnumeric()]
        if len(digits) > 0:
            sub = str(np.max(np.asarray(sub_dirs).astype('uint8')) + 1)
        else:
            sub = '1'
        path = os.path.join(path, sub)
        os.makedirs(path)
    print('Logging to:%s' % path)
    return path


def build_targeted_dataset(X_test, Y_test, indices, nb_classes, img_rows, img_cols, img_channels):
    """
    Build a dataset for targeted attacks, each source image is repeated nb_classes -1
    times, and target labels are assigned that do not overlap with true label. 
    :param X_test: clean source images
    :param Y_test: true labels for X_test
    :param indices: indices of source samples to use
    :param nb_classes: number of classes in classification problem
    :param img_rows: number of pixels along rows of image
    :param img_cols: number of pixels along columns of image
    """

    nb_samples = len(indices)
    nb_target_classes = nb_classes - 1
    X = X_test[indices]
    Y = Y_test[indices]

    adv_inputs = np.array(
        [[instance] * nb_target_classes for
         instance in X], dtype=np.float32)
    adv_inputs = adv_inputs.reshape(
        (nb_samples * nb_target_classes, img_rows, img_cols, img_channels))

    true_labels = np.array(
        [[instance] * nb_target_classes for
         instance in Y], dtype=np.float32)
    true_labels = true_labels.reshape(
        nb_samples * nb_target_classes, nb_classes)

    target_labels = np.zeros((nb_samples * nb_target_classes, nb_classes))

    for n in range(nb_samples):
        one_hot = np.zeros((nb_target_classes, nb_classes))
        one_hot[np.arange(nb_target_classes), np.arange(nb_classes)
                != np.argmax(Y[n])] = 1.0
        start = n * nb_target_classes
        end = start + nb_target_classes
        target_labels[start:end] = one_hot

    return adv_inputs, true_labels, target_labels


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
