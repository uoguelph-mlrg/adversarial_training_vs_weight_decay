import tensorflow as tf
import matplotlib.pyplot as plt

THRESH = 1e-9


def quantize(x, k):
    n = float(2**k - 1)
    G = tf.get_default_graph()
    with G.gradient_override_map({"Round": "Identity"}):
        return tf.round(x * n) / n


def fw(x, bitW):
    if bitW == 32:
        return x
    if bitW == 1:   # BWN
        G = tf.get_default_graph()
        with G.gradient_override_map({"Sign": "Identity"}):
            E = tf.stop_gradient(tf.reduce_mean(tf.abs(x)))
            return tf.sign(x / E) * E
    x = tf.tanh(x)
    x = x / tf.reduce_max(tf.abs(x)) * 0.5 + 0.5
    return 2 * quantize(x, bitW) - 1


def prune(data):
    for i in range(data.shape[0]):
        if -THRESH < data[i] and data[i] < THRESH:
            data[i] = 0.
    return data


def prune_tensor(tensor):
    t_min = tf.where(-THRESH < tensor, tf.constant(0.,
                                                   shape=tensor.shape), tensor)
    t_max = tf.where(tensor < THRESH, tf.constant(
        0., shape=tensor.shape), tensor)
    return tf.assign(tensor, t_min + t_max)


def restore_model(sess, restorer, ckpt_path):
    model_tokens = ckpt_path.split('/')
    if 'model' in model_tokens[-1]:
        restorer.restore(sess, ckpt_path)
    else:
        restorer.restore(sess, tf.train.latest_checkpoint(ckpt_path))
    print('Restored model')


def init_figure():
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    return ax


def disp_data(data):
    ax = init_figure()
    ax.imshow(data.reshape(28, 28), cmap='gray')
    plt.tight_layout()
    plt.show()


def disp_index_data(data, index):
    ax = init_figure()
    ax.imshow(data[index].reshape(28, 28), cmap='gray')
    plt.tight_layout()
    plt.show()
