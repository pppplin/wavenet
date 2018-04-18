from __future__ import division

import tensorflow as tf


def create_adam_optimizer(learning_rate, momentum):
    return tf.train.AdamOptimizer(learning_rate=learning_rate,
                                  epsilon=1e-4)


def create_sgd_optimizer(learning_rate, momentum):
    return tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                      momentum=momentum)


def create_rmsprop_optimizer(learning_rate, momentum):
    return tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                     momentum=momentum,
                                     epsilon=1e-5)


optimizer_factory = {'adam': create_adam_optimizer,
                     'sgd': create_sgd_optimizer,
                     'rmsprop': create_rmsprop_optimizer}


def time_to_batch(value, dilation, velocity=False, name=None):
    with tf.name_scope('time_to_batch'):
        shape = tf.shape(value)
        pad_elements = dilation - 1 - (shape[1] + dilation - 1) % dilation
        if velocity:
            padded = tf.pad(value, [[0, 0], [0, pad_elements],
                [0, 0], [0, 0]])
            reshaped = tf.reshape(padded, [-1, dilation, 2, shape[-1]])
            transposed = tf.transpose(reshaped, perm=[1, 0, 2, 3])
            return tf.reshape(transposed, [shape[0]*dilation, -1, 2, shape[-1]])
        else:
            padded = tf.pad(value, [[0, 0], [0, pad_elements], [0, 0]])
            reshaped = tf.reshape(padded, [-1, dilation, shape[2]])
            transposed = tf.transpose(reshaped, perm=[1, 0, 2])
            return tf.reshape(transposed, [shape[0] * dilation, -1, shape[2]])


def batch_to_time(value, dilation, velocity=False, name=None):
    with tf.name_scope('batch_to_time'):
        shape = tf.shape(value)
        if velocity:
            prepared = tf.reshape(value, [dilation, -1, 2, shape[-1]])
            transposed = tf.transpose(prepared, perm=[1, 0, 2, 3])
            return tf.reshape(transposed,
                            [tf.div(shape[0], dilation), -1, 2, shape[-1]])
        else:
            prepared = tf.reshape(value, [dilation, -1, shape[2]])
            transposed = tf.transpose(prepared, perm=[1, 0, 2])
            return tf.reshape(transposed,
                          [tf.div(shape[0], dilation), -1, shape[2]])


def causal_conv(value, filter_, dilation, velocity_input=False, name='causal_conv'):
    with tf.name_scope(name):
        filter_width = tf.shape(filter_)[0]
        if dilation > 1:
            if velocity_input:
                transformed = time_to_batch(value, dilation, velocity=True)
                conv = tf.nn.conv2d(transformed, filter_, strides=[1,1,1,1], padding='VALID')
                restored = batch_to_time(conv, dilation, velocity=True)
            else:
                transformed = time_to_batch(value, dilation)
                conv = tf.nn.conv1d(transformed, filter_, stride=1,
                                padding='VALID')
                restored = batch_to_time(conv, dilation)
        else:
            if velocity_input:
                restored = tf.nn.conv2d(value, filter_, strides=[1,1,1,1], padding='VALID')
            else:
                restored = tf.nn.conv1d(value, filter_, stride=1, padding='VALID')
        # Remove excess elements at the end.
        out_width = tf.shape(value)[1] - (filter_width - 1) * dilation
        if velocity_input:
            result = tf.slice(restored,
                        [0, 0, 0, 0],
                        [-1, out_width, -1, -1])
        else:
            result = tf.slice(restored,
                          [0, 0, 0],
                          [-1, out_width, -1])
        return result

def mu_law_encode(audio, quantization_channels):
    '''Quantizes waveform amplitudes.'''
    with tf.name_scope('encode'):
        mu = tf.to_float(quantization_channels - 1)
        # Perform mu-law companding transformation (ITU-T, 1988).
        # Minimum operation is here to deal with rare large amplitudes caused
        # by resampling.
        safe_audio_abs = tf.minimum(tf.abs(audio), 1.0)
        magnitude = tf.log1p(mu * safe_audio_abs) / tf.log1p(mu)
        signal = tf.sign(audio) * magnitude
        # Quantize signal to the specified number of levels.
        return tf.to_int32((signal + 1) / 2 * mu + 0.5)


def mu_law_decode(output, quantization_channels):
    '''Recovers waveform from quantized values.'''
    with tf.name_scope('decode'):
        mu = quantization_channels - 1
        # Map values back to [-1, 1].
        signal = 2 * (tf.to_float(output) / mu) - 1
        # Perform inverse of mu-law transformation.
        magnitude = (1 / mu) * ((1 + mu)**abs(signal) - 1)
        return tf.sign(signal) * magnitude
