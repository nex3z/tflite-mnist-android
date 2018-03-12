import tensorflow as tf

IMAGE_SIZE = 28
IMAGE_CHANNEL_NUM = 1

OUTPUT_SIZE = 10

CONV_1_SIZE = 6
CONV_1_DEPTH = 6

CONV_2_SIZE = 5
CONV_2_DEPTH = 12

CONV_3_SIZE = 4
CONV_3_DEPTH = 24

FC_SIZE = 200


def get_weight(shape):
    return tf.get_variable("weight", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))


def get_bias(shape):
    return tf.get_variable("bias", shape, initializer=tf.constant_initializer(0.0))


def conv2d(input_tensor, weight, stride):
    return tf.nn.conv2d(input_tensor, weight, strides=[1, stride, stride, 1], padding="SAME")


def inference(input_tensor, regularizer=None):
    with tf.variable_scope("layer_1_conv"):
        conv_1_weight = get_weight([CONV_1_SIZE, CONV_1_SIZE, IMAGE_CHANNEL_NUM, CONV_1_DEPTH])
        conv_1_bias = get_bias([CONV_1_DEPTH])
        conv_1 = conv2d(input_tensor, conv_1_weight, stride=1)
        conv_1_activation = tf.nn.relu(tf.nn.bias_add(conv_1, conv_1_bias))
    with tf.variable_scope("layer_2_conv"):
        conv_2_weight = get_weight([CONV_2_SIZE, CONV_2_SIZE, CONV_1_DEPTH, CONV_2_DEPTH])
        conv_2_bias = get_bias([CONV_2_DEPTH])
        conv_2 = conv2d(conv_1_activation, conv_2_weight, stride=2)
        conv_2_activation = tf.nn.relu(tf.nn.bias_add(conv_2, conv_2_bias))
    with tf.variable_scope("layer_3_conv"):
        conv_3_weight = get_weight([CONV_3_SIZE, CONV_3_SIZE, CONV_2_DEPTH, CONV_3_DEPTH])
        conv_3_bias = get_bias([CONV_3_DEPTH])
        conv_3 = conv2d(conv_2_activation, conv_3_weight, stride=2)
        conv_3_activation = tf.nn.relu(tf.nn.bias_add(conv_3, conv_3_bias))
        shape = conv_3_activation.get_shape().as_list()
        nodes = shape[1] * shape[2] * shape[3]
        conv_3_activation_reshaped = tf.reshape(conv_3_activation, [-1, nodes])
    with tf.variable_scope("layer_4_fc"):
        w4 = get_weight([nodes, FC_SIZE])
        if regularizer is not None:
            tf.add_to_collection("losses", regularizer(w4))
        b4 = get_bias([FC_SIZE])
        a4 = tf.nn.relu(tf.matmul(conv_3_activation_reshaped, w4) + b4)
    with tf.variable_scope("layer_5_fc"):
        w5 = get_weight([FC_SIZE, OUTPUT_SIZE])
        if regularizer is not None:
            tf.add_to_collection("losses", regularizer(w5))
        b5 = get_bias([OUTPUT_SIZE])
        logits = tf.matmul(a4, w5) + b5
    return logits