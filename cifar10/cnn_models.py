"""
A pure TensorFlow implementation of a neural network. This can be
used as a drop-in replacement for a Keras model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
from cleverhans.model import Model

BN_EPSILON = 1e-5


@tf.RegisterGradient("QuantizeGrad")
def quantize_grad(op, grad):
    return tf.clip_by_value(tf.identity(grad), -1, 1)


def hard_sigmoid(x):
    return tf.cast(tf.clip_by_value((x + 1.) / 2., 0., 1.), tf.float32)


class MLP(Model):
    """
    An example of a bare bones multilayer perceptron (MLP) class.
    """

    def __init__(self, layers, input_shape):
        super(MLP, self).__init__()

        self.layer_names = []
        self.layers = layers
        self.input_shape = input_shape
        if isinstance(layers[-1], Softmax):
            layers[-1].name = 'probs'
            layers[-2].name = 'logits'
        else:
            layers[-1].name = 'logits'
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'name'):
                name = layer.name
            else:
                name = layer.__class__.__name__ + str(i)
            self.layer_names.append(name)

            layer.set_input_shape(input_shape)
            input_shape = layer.get_output_shape()
        print(self.layer_names)
        self.n_params = np.sum([np.prod(v.shape)
                                for v in tf.trainable_variables()])
        print(self.n_params)

    def fprop(self, x, set_ref=False):
        states = []
        for layer in self.layers:
            if set_ref:
                layer.ref = x
            x = layer.fprop(x)
            assert x is not None
            states.append(x)
        states = dict(zip(self.get_layer_names(), states))
        return states


class Layer(object):

    def get_output_shape(self):
        return self.output_shape


class SimpleLinear(Layer):

    def __init__(self, num_hid):
        self.num_hid = num_hid

    def set_input_shape(self, input_shape, reuse):
        batch_size, dim = input_shape
        self.input_shape = [batch_size, dim]
        self.output_shape = [batch_size, self.num_hid]
        init = tf.random_normal([dim, self.num_hid], dtype=tf.float32)
        init = init / tf.sqrt(1e-7 + tf.reduce_sum(tf.square(init), axis=0,
                                                   keep_dims=True))
        self.W = tf.Variable(init)
        self.b = tf.Variable(np.zeros((self.num_hid,)).astype('float32'))

    def fprop(self, x, reuse):
        return tf.matmul(x, self.W) + self.b


class Linear(Layer):

    # def __init__(self, num_hid, scope_name):
    def __init__(self, num_hid, detail):
        self.__dict__.update(locals())
        # self.num_hid = num_hid

    def set_input_shape(self, input_shape):

        # with tf.variable_scope(self.scope_name+ 'init', reuse): # this works
        # with black box, but now can't load checkpoints from wb
        # this works with white-box
        with tf.variable_scope(self.detail + '_init', reuse=tf.AUTO_REUSE):

            batch_size, dim = input_shape
            self.input_shape = [batch_size, dim]
            self.output_shape = [batch_size, self.num_hid]
            init = tf.random_normal([dim, self.num_hid], dtype=tf.float32)
            init = init / tf.sqrt(1e-7 + tf.reduce_sum(tf.square(init), axis=0,
                                                       keep_dims=True))
            self.W = tf.get_variable(
                "W", initializer=init)

            # Add L1 and L2 weight decay penalties
            self.l1_p = tf.reduce_mean(tf.abs(self.W))
            self.l2_p = tf.nn.l2_loss(self.W)

            W_summ = tf.summary.histogram('W', values=self.W)

    def fprop(self, x):

        # with tf.variable_scope(self.scope_name + '_fprop', reuse):
        # this works with white-box
        with tf.variable_scope(self.detail + '_fprop', reuse=tf.AUTO_REUSE):

            a = tf.matmul(x, self.W)  # + self.b
            a_u, a_v = tf.nn.moments(tf.abs(a), axes=[0], keep_dims=False)
            a_summ = tf.summary.histogram('a', values=a)
            a_u_summ = tf.summary.scalar("a_u", tf.reduce_mean(a_u))
            a_v_summ = tf.summary.scalar("a_v", tf.reduce_mean(a_v))

            return a


class Conv2D(Layer):

    def __init__(self, binary, output_channels, kernel_shape, strides, padding, scope_name):
        self.__dict__.update(locals())
        self.G = tf.get_default_graph()
        del self.self

    def quantize(self, x):
        with self.G.gradient_override_map({"Sign": "QuantizeGrad"}):
            return tf.sign(x)

    def set_input_shape(self, input_shape):

        batch_size, rows, cols, input_channels = input_shape
        kernel_shape = tuple(self.kernel_shape) + (input_channels,
                                                   self.output_channels)
        assert len(kernel_shape) == 4
        assert all(isinstance(e, int) for e in kernel_shape), kernel_shape

        with tf.variable_scope(self.scope_name + '_init', reuse=tf.AUTO_REUSE):

            if self.binary:
                init = tf.truncated_normal(
                    kernel_shape, stddev=0.2, dtype=tf.float32)
            else:
                init = tf.truncated_normal(
                    kernel_shape, stddev=0.1, dtype=tf.float32)
                init = init / tf.sqrt(1e-7 + tf.reduce_sum(tf.square(init),
                                                           axis=(0, 1, 2)))
            self.real_kernels = tf.get_variable("k", initializer=init)

            # Add L1 and L2 weight decay penalties
            self.l1_p = tf.reduce_mean(tf.abs(self.real_kernels))
            self.l2_p = tf.nn.l2_loss(self.real_kernels)

            k_summ = tf.summary.histogram(
                name="k", values=self.real_kernels)

            if self.binary:
                self.kernels = self.quantize(self.real_kernels)
                k_bin_summ = tf.summary.histogram(
                    name="k_bin", values=self.kernels)

            orig_input_batch_size = input_shape[0]
            input_shape = list(input_shape)
            input_shape[0] = 1
            dummy_batch = tf.zeros(input_shape)
            dummy_output = self.fprop(dummy_batch)
            output_shape = [int(e) for e in dummy_output.get_shape()]
            output_shape[0] = 1
            self.output_shape = tuple(output_shape)

    def fprop(self, x):

        # need variable_scope here because the batch_norm layer creates
        # variables internally
        with tf.variable_scope(self.scope_name + '_fprop', reuse=tf.AUTO_REUSE) as scope:

            if self.binary:
                # center input to quantization op
                '''
                x = tf.contrib.layers.batch_norm(
                    x, epsilon=BN_EPSILON, is_training=self.phase,
                    reuse=reuse, scope=scope)
                '''
                x = self.quantize(x)
                a = tf.nn.conv2d(x, self.kernels, (1,) +
                                 tuple(self.strides) + (1,), self.padding)
            else:
                a = tf.nn.conv2d(x, self.real_kernels, (1,) +
                                 tuple(self.strides) + (1,), self.padding)
            a_u, a_v = tf.nn.moments(tf.abs(a), axes=[0], keep_dims=False)
            a_summ = tf.summary.histogram('a', values=a)
            a_u_summ = tf.summary.scalar("a_u", tf.reduce_mean(a_u))
            a_v_summ = tf.summary.scalar("a_v", tf.reduce_mean(a_v))

            return a


class ReLU(Layer):

    def __init__(self):
        pass

    def set_input_shape(self, shape):
        self.input_shape = shape
        self.output_shape = shape

    def get_output_shape(self):
        return self.output_shape

    def fprop(self, x):
        return tf.nn.relu(x)


class Softmax(Layer):

    def __init__(self):
        pass

    def set_input_shape(self, shape):
        self.input_shape = shape
        self.output_shape = shape

    def fprop(self, x):
        return tf.nn.softmax(x)


class Flatten(Layer):

    def __init__(self):
        pass

    def set_input_shape(self, shape):
        self.input_shape = shape
        output_width = 1
        for factor in shape[1:]:
            output_width *= factor
        self.output_width = output_width
        self.output_shape = [None, output_width]

    def fprop(self, x):
        return tf.reshape(x, [-1, self.output_width])


######################### full-precision #########################


def make_basic_cnn(detail, nb_filters=64, nb_classes=10,
                   input_shape=(None, 28, 28, 1)):
    layers = [Conv2D(False, nb_filters, (8, 8), (2, 2), "SAME", detail + 'conv1'),
              ReLU(),
              Conv2D(False, nb_filters * 2, (6, 6),
                     (2, 2), "VALID", detail + 'conv2'),
              ReLU(),
              Conv2D(False, nb_filters * 2, (5, 5), (1, 1),
                     "VALID", detail + 'conv3'),
              ReLU(),
              Flatten(),
              Linear(nb_classes, detail),
              Softmax()]

    model = MLP(layers, input_shape)
    print('Finished making basic cnn')
    return model
