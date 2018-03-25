from argparse import ArgumentParser

import tensorflow as tf
from tensorflow.python.framework import graph_util
import train

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
    # Create a model to classify single image
    x_single = tf.placeholder(tf.float32, [1, train.IMAGE_SIZE, train.IMAGE_SIZE, train.IMAGE_CHANNEL_NUM],
                              name="input_single")
    y_single = train.inference(x_single)
    output_single = tf.identity(tf.nn.softmax(y_single, axis=1), name="output_single")

    with tf.Session() as sess:
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(options.model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("Restoring %s" % ckpt.model_checkpoint_path)
            # Restore the model trained by train.py
            saver.restore(sess, ckpt.model_checkpoint_path)
            graph_def = tf.get_default_graph().as_graph_def()
            # Freeze the graph
            output_graph = graph_util.convert_variables_to_constants(sess, graph_def, ["output_single"])
            # The input type and shape of the converted model is inferred from the input_tensors argument
            tflite_model = tf.contrib.lite.toco_convert(
                output_graph, input_tensors=[x_single], output_tensors=[output_single])
            with open(options.output_file, "wb") as f:
                f.write(tflite_model)
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
