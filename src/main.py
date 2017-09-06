import numpy as np
import tensorflow as tf
import cv2
import os
import sys

from inception.inception_v4 import inception_v4
from inception import inception_utils
slim = tf.contrib.slim

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CURRENT_PATH)

def get_data_list(dir_path, size):
    """
    
    :param dir_path: images directory path
    :param size: image target size 
    :return: list of images, in a uniform size of size x size, anda list of their full path
    """

    images = os.listdir(dir_path)
    img_list = []
    img_path_list = []
    for fn in images:
        cur_fn = os.path.abspath(os.path.join(dir_path, fn))
        img = cv2.imread(cur_fn, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 3:
            img_list.append(fit_image(img, size))
            img_path_list.append(cur_fn)
    return img_list, img_path_list

def fit_image(rgb_img, size):
    """
    given an RGB image, return the new image in the given size, 
    using crop from the center and resize
    
    :param img: the image that should be resized
    :param size: image target size 
    :return: image size of size x size
    """
    size_x = rgb_img.shape[1]
    size_y = rgb_img.shape[0]
    center = (np.uint32(size_y/2), np.uint32(size_x/2))

    if size_x < size_y:
        cropped = rgb_img[(center[0]-center[1]):(center[0]+center[1]+(size_x%2)), :]
    else:
        cropped = rgb_img[:, (center[1] - center[0]):(center[0] + center[1] + (size_x % 2))]

    resized = cv2.resize(cropped, (size, size))
    return resized

def create_image_batches(img_list, img_path_list, batch_size):
    """
    
    :param img_list: list of images
    :param batch_size: size of wanted batch
    :return: list of batches of images, where each batch maximum size is batch_size
    """
    batch_num = np.uint32(np.ceil(len(img_list)/batch_size))
    batches = []
    paths_batches = []
    for i in range(batch_num):
        batches.append(img_list[i*batch_size: ((i+1)*batch_size)])
        paths_batches.append(img_path_list[i * batch_size: ((i + 1) * batch_size)])
    return batches, paths_batches

def run_model(img_list, img_path_list):
    """
    Gets image batches, run the pretrained inception_v4 on all the
    images, and return batches of descriptors (one for each image)
    :param batches: image batches 
    :return: descriptor batches
    """
    batch_size = 50
    height, width = 299, 299

    checkpoint_path = r"..\trained\inception_v4.ckpt"

    with tf.Graph().as_default():
        tf_global_step = slim.get_or_create_global_step()

        arg_scope = inception_utils.inception_arg_scope()

        # eval_inputs = tf.random_uniform((batch_size, height, width, 3))
        inputs = tf.placeholder(tf.float32, (None, height, width, 3))

        with slim.arg_scope(arg_scope):
            logits, end_points = inception_v4(inputs, is_training=False)
        predictions = tf.argmax(logits, 1)

        # Create a saver.
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, checkpoint_path)

            # output = sess.run(predictions)
            batches, batches_path_list = create_image_batches(img_list, img_path_list, batch_size)
            descriptor_batches = []
            for batch_num in range(len(batches)):
                images = batches[batch_num]
                descriptor = sess.run(end_points['PreLogitsFlatten'], feed_dict={inputs: images})
                descriptor_batches.append(descriptor)
                print(descriptor)
    return descriptor_batches, batches_path_list


if __name__ == "__main__":
    # PARAMS:
    img_size = 299

    images_dir = os.path.join(CURRENT_PATH, "..", "data_set", "Zuriel vila")
    # Get image list
    img_list, img_path_list = get_data_list(images_dir, img_size)
    descriptors, descriptors_path_list = run_model(img_list, img_path_list)
    print("")