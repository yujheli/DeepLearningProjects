import tensorflow as tf

def fullyConnectedLayer(input_feature, output_dim=None, name="fullyConnectedLayer"):
    with tf.variable_scope(name):
        weights = tf.get_variable(name="weights", shape=[input_feature.get_shape().as_list()[1], output_dim], dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
        biases = tf.get_variable(name="biases", shape=[output_dim], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        return tf.matmul(input_feature, weights) + biases

def convolutionLayer(input_feature, filter_shape=None, stride_shape=None, name="convolutionLayer"):
    with tf.variable_scope(name):
        weights = tf.get_variable(name="weights", shape=filter_shape, dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.02))
        biases = tf.get_variable(name="biases", shape=[filter_shape[-1]], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        z = tf.nn.conv2d(input_feature, weights, strides=stride_shape, padding="SAME")
        z = tf.reshape(tf.nn.bias_add(z, biases), z.get_shape())
        return z

def transConvolutionLayer(input_feature, filter_shape=None, stride_shape=None, output_shape=None, name="transConvolutionLayer"):
    with tf.variable_scope(name):
        weights = tf.get_variable(name="weights", shape=filter_shape, dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
        biases = tf.get_variable(name="biases", shape=[output_shape[-1]], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        z = tf.nn.conv2d_transpose(input_feature, weights, output_shape=output_shape, strides=stride_shape, padding="SAME")
        z = tf.reshape(tf.nn.bias_add(z, biases), z.get_shape())
        return z

def batchNormLayer(input_feature, is_training, name="batchNormLayer"):
    with tf.variable_scope(name):
        return tf.contrib.layers.batch_norm(input_feature, decay=0.9, center=True, scale=True, epsilon=1e-5, updates_collections=None, is_training=is_training)
