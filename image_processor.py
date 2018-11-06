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
    binary_local = cv2.adaptiveThreshold(
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


def get_next_notch_center(im, clf=None):
    # разбиваем изображение на отдельные figures
    nb_components, output, stats, centroids = \
        cv2.connectedComponentsWithStats(np.array(im, dtype=np.uint8), connectivity=4)

    # отрезаем 1й элемент - фон
    stats = stats[1:]
    centroids = centroids[1:]

    # координаты центра фото
    c_y, c_x = [s / 2 for s in im.shape]

    # определяем notches среди figures
    # labels = clf.predict(stats[:, 2:5])

    # вычисляем дистанцию от центра каждой figure до центра экрана
    dists = [((c_x - x) ** 2 + (c_y - y) ** 2) ** .5 for x, y in centroids]

    # dists = [d for d, l in zip(dists, labels) if l]
    # centroids = [c for c, l in zip(centroids, labels) if l]

    if len(dists) == 0:
        raise LackOfNotch()

    nh_ix = dists.index(min(dists))

    x, y = centroids[nh_ix]
    x, y = x - c_x, y - c_y

    return (x, y)


def parse_filename(fname):
    fname = fname.split('.')[0]
    x, y = [float(i.replace(',', '.')) for i in fname.split('x')]
    return x, y