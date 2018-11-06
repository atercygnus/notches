import sys
import os

import cv2
import numpy as np

from  point_organizer import *
from  image_processor import *

from skimage import io

if len(sys.argv) > 1:
    path = sys.argv[1]
    if path[-1] != '\\':
        path += '\\'
else:
    path = os.getcwd() + '\\photo\\'

if len(sys.argv) > 2:
    pix_mm = float(sys.argv[2])
else:
    pix_mm = 1275 / 8

impaths = [f for f in listdir(path) if isfile(join(path, f))]

notch_center_x = []
notch_center_y = []

im_center_x = []
im_center_y = []

for n, impath in enumerate(impaths):
    print("processing " + impath + " " + "({} of {})".format(n + 1, len(impaths)))
    image = cv2.imread(path + impath, cv2.IMREAD_GRAYSCALE)
    bimage = binarize(image)

    try:
        bpath = os.getcwd() + '\\binarized\\'
        os.mkdir(bpath)
    except OSError:
        pass

    io.imsave(os.getcwd() + '\\binarized\\' + impath, bimage)

    nh_center_x, nh_center_y = get_next_notch_center(bimage)

    notch_center_x.append(nh_center_x)
    notch_center_y.append(nh_center_y)

    im_x, im_y = parse_filename(impath)
    im_center_x.append(im_x)
    im_center_y.append(im_y)

notch_center_x = np.array(notch_center_x)
notch_center_y = np.array(notch_center_y)
im_center_x = np.array(im_center_x)
im_center_y = np.array(im_center_y)

notch_center_x=np.array([x/pix_mm for x in notch_center_x])
notch_center_y=np.array([y/pix_mm for y in notch_center_y])

notch_x = np.array([x1+x2 for x1, x2 in zip(notch_center_x, im_center_x)])
notch_y = np.array([y1+y2 for y1, y2 in zip(notch_center_y, im_center_y)])
notch = np.array(list(zip(notch_x, notch_y)))

# ----

delta = 3

right_ix = np.array([i for i, x in enumerate(notch_x) if max(notch_x)-delta <= x <= max(notch_x)])
right = notch[right_ix]

left_ix = np.array([i for i, x in enumerate(notch_x) if min(notch_x) <= x <= min(notch_x) + delta])
left = notch[left_ix]

top_ix = np.array([i for i, x in enumerate(notch_y) if max(notch_y)-delta <= x <= max(notch_y)])
top = notch[top_ix]

bottom_ix = np.array([i for i, x in enumerate(notch_y) if min(notch_y) <= x <= min(notch_y) + delta])
bottom = notch[bottom_ix]

top_x = notch_x[top_ix]
top_y = notch_y[top_ix]

bottom_x = notch_x[bottom_ix]
bottom_y = notch_y[bottom_ix]

left_x = notch_x[left_ix]
left_y = notch_y[left_ix]

right_x = notch_x[right_ix]
right_y = notch_y[right_ix]

bottom_top = []
for pair in get_pairs(top_x.tolist(), bottom_x.tolist()):
    mean_top_y = np.mean(top_y)
    mean_bottom_y = np.mean(bottom_y)
    pta = (pair[0], mean_top_y)
    ptb = (pair[1], mean_bottom_y)
    bottom_top.append([pta, ptb])

left_right = []
for pair in get_pairs(left_y.tolist(), right_y.tolist()):
    mean_left_x = np.mean(left_x)
    mean_right_x = np.mean(right_x)
    pta = (mean_left_x, pair[0])
    ptb = (mean_right_x, pair[1])
    left_right.append([pta, ptb])

from itertools import product

grid = []
for ln1, ln2 in product(left_right, bottom_top):
    k1, b1 = get_line(*ln1)
    k2, b2 = get_line(*ln2)
    x, y = get_intersection(k1, b1, k2, b2)
    grid.append([round(x, 2), round(y, 2)])

f = open("grid.txt", "w")
for pt in grid:
    f.write("%s\n" % pt)
f.close()
