from argparse import ArgumentParser

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def main():
    parser = build_parser()
    options = parser.parse_args()
    mnist_data = input_data.read_data_sets("./data", one_hot=True, reshape=False)
    test(mnist_data, options)


def test(mnist_data, options):
    checkpoint = tf.train.latest_checkpoint(options.model_dir)
    meta = checkpoint + ".meta"
    print("Loading %s" % meta)
    saver = tf.train.import_meta_graph(meta)
    with tf.Session() as sess:
        print("Restoring %s" % checkpoint)
        saver.restore(sess, checkpoint)
        accuracy = tf.get_default_graph().get_tensor_by_name("accuracy:0")
        test_accuracy = sess.run(accuracy, feed_dict={"x:0": mnist_data.test.images, "y_:0": mnist_data.test.labels})
        print("Test accuracy: %g" % test_accuracy)


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--model_dir', dest='model_dir', default='./saved_model',
                        help='directory where checkpoints are saved')
    return parser


if __name__ == '__main__':
    main()
