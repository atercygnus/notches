import os
import matplotlib.pyplot as plt
import numpy as np

import cv2
from cv2 import dilate
from cv2 import adaptiveThreshold

from os import listdir
from os.path import isfile, join

from skimage import io

from sklearn.externals import joblib

def cv_remove_small_objects(im, min_size):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(im, connectivity=8)
    sizes = stats[1:, -1];
    nb_components = nb_components - 1

    res = np.zeros((output.shape))
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            res[output == i + 1] = 1

    return res


def binarize(im):
    binary_local = im = cv2.adaptiveThreshold(
        src=np.array(cv2.medianBlur(im, 5), dtype=np.uint8),
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=505,
        C=25)

    cleaned = cv_remove_small_objects(binary_local, 250)

    strel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))
    cleaned = cv2.dilate(np.array(cleaned, dtype=np.uint8), strel)

    cleaned = cv_remove_small_objects(cleaned, 15000)

    return cleaned


def get_next_notch_center(im):
    nb_components, output, stats, centroids = \
        cv2.connectedComponentsWithStats(np.array(im, dtype=np.uint8), connectivity=4)

    stats = np.array(stats)
    stats = stats[:, 2:5]

    c_y, c_x = [s / 2 for s in im.shape]
    dists = [((c_x - x) ** 2 + (c_y - y) ** 2) ** .5 for x, y in centroids[1:]]
    nh_ix = dists.index(min(dists)) + 1

    x, y = centroids[nh_ix]
    x, y = x - c_x, y - c_y

    return (x, y)


def parse_filename(fname):
    fname = fname.split('.')[0]
    x, y = [float(i.replace(',', '.')) for i in fname.split('x')]
    return x, y