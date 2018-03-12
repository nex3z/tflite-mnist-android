from argparse import ArgumentParser

import tensorflow as tf
from tensorflow.python.framework import graph_util
from model import mnist as model

# A workaround to fix an import issue
# see https://github.com/tensorflow/tensorflow/issues/15410#issuecomment-352189481
import subprocess
import tempfile
tf.contrib.lite.tempfile = tempfile
tf.contrib.lite.subprocess = subprocess


def main():
    parser = build_parser()
    options = parser.parse_args()
    convert(options)


def convert(options):
    x_single = tf.placeholder(tf.float32, [1, model.IMAGE_SIZE, model.IMAGE_SIZE, model.IMAGE_CHANNEL_NUM],
                              name="input_single")
    y_single = model.inference(x_single)
    output_single = tf.identity(tf.nn.softmax(y_single, axis=1), name="output_single")

    with tf.Session() as sess:
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(options.model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("Restoring %s" % ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            graph_def = tf.get_default_graph().as_graph_def()
            output_graph = graph_util.convert_variables_to_constants(sess, graph_def, ["output_single"])
            tflite_model = tf.contrib.lite.toco_convert(output_graph, [x_single], [output_single])
            open(options.output_file, "wb").write(tflite_model)
        else:
            print("Checkpoint not found")


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--model_dir', dest='model_dir', default='./saved_model',
                        help='directory where checkpoints are saved')
    parser.add_argument('--output_file', dest='output_file', default='./mnist.tflite',
                        help='destination to write the converted file')
    return parser


if __name__ == '__main__':
    main()
