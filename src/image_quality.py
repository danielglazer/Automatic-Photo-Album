import cv2
import itertools
import numpy as np
import sys


def image_global_contrast(image):
    """
    :param image: A rgb image 
    :return: Image contrast
    """
    avg = np.mean(image)
    variance = np.mean(np.square(image - avg))
    return np.sqrt(np.sqrt(sum(np.square(variance))))


def image_resolution(image):
    """
    :param image: An image
    :return: The number of pixels in the image
    """
    return image.shape[0] * image.shape[1]


def image_noise(image):
    """
   :param image: An image
   :return: How noisy the image is 
   """
    pass


def contrast_sort(image_list):
    """
    :param image_list: List of images
    :return: An array of indices of the images sorted by contrast 
    """
    graded_image_list = image_global_contrast(image_list)
    return np.argsort(graded_image_list)


def image_perceived_sharpness(image):
    """
    :param image: An rgb image
    :return: The image perceived sharpness
    """
    # Perceived sharpness is a combination of both resolution and acutance

    # The acutance of an image is a vector field
    # The accutance is the mean value of the Gradient Filter
    #TODO: check if this is working for rgb image
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    accutance = np.mean(laplacian)
    # Resolution is the number of pixels in the image
    resolution = image.size
    return accutance, resolution



