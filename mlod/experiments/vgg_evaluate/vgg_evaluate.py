import argparse
import datetime
import sys

import cv2
import tensorflow as tf

import mlod
from mlod.datasets.imagenet import imagenet
from mlod.experiments.vgg_evaluate import vgg_network

# Flags set by argparse
FLAGS = None


def fill_feed_dict(input_pl):

    image_name = 'cat.jpg'
    # image_name = 'school_bus.jpg'
    # image_name = 'table.jpg'

    cv2_bgr_image = cv2.imread(mlod.root_dir() + '/tests/images/' + image_name)

    # BGR -> RGB
    rgb_image = cv2_bgr_image[..., :: -1]

    feed_dict = {
        # input_pl: bev_image,
        input_pl: rgb_image,
    }

    return feed_dict


def main(_):

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():

        with tf.name_scope('input'):
            # Placeholder for image input, to be filled in with a feed_dict
            image_placeholder = tf.placeholder(tf.float32, (None, None, 3))

            image_summary = tf.expand_dims(image_placeholder, axis=0)
            tf.summary.image("image", image_summary, max_outputs=5)

        # Create the session
        sess = tf.InteractiveSession()

        logits, probabilities = vgg_network.inference(sess, image_placeholder)

        # Merge all the summaries and write them out to FLAGS.log_dir
        summary_merged = tf.summary.merge_all()

        # Create unique folder name using datetime for summary writer
        datetime_str = str(datetime.datetime.now())
        summary_writer = tf.summary.FileWriter(
            FLAGS.log_dir + '/' + datetime_str, sess.graph)

        feed_dict = fill_feed_dict(image_placeholder)

        # Run the inference
        summary, probabilities = sess.run(
            [summary_merged, probabilities], feed_dict=feed_dict)

        probabilities = probabilities[0, 0:]
        sorted_indices = [
            i[0]
            for i in sorted(enumerate(-probabilities), key=lambda x: x[1])
        ]

        # Write the summary
        summary_writer.add_summary(summary)

        # Close the summary writers
        summary_writer.close()

        names = imagenet.create_readable_names_for_imagenet_labels()
        for i in range(5):
            index = sorted_indices[i]
            # Shift the index of a class name by one.
            print('Probability %0.2f%% => [%s]' % (probabilities[index] * 100,
                                                   names[index + 1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--fake_data',
        nargs='?',
        const=True,
        type=bool,
        default=False,
        help='If true, uses fake data for unit testing.')
    parser.add_argument(
        '--max_steps',
        type=int,
        default=1000,
        help='Number of steps to run trainer.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Initial learning rate')
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.9,
        help='Keep probability for training dropout.')
    # parser.add_argument(
    #     '--data_dir',
    #     type=str,
    #     default='/tmp/mlod/vgg16/input_data',
    #     help='Directory for storing input data')
    parser.add_argument(
        '--log_dir',
        type=str,
        default=mlod.root_dir() + '/logs/vgg16',
        help='Summaries log directory')

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
