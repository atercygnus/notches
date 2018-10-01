import statistics as stat
import numpy as np


# каждой точке из left_pts ищет пару из right_pts
def left_to_right(left_pts, right_pts):
    pairs = []
    min_dists = []
    for a in left_pts:
        dists = [abs(a - b) for b in right_pts]
        min_dists.append(min(dists))
        pairs.append([a, right_pts[dists.index(min(dists))]])
    min_dists_mean = stat.mean(min_dists)
    min_dists_std = stat.stdev(min_dists)

    #
    pairs = [p if m < min_dists_mean + 2 * min_dists_std else [p[0], None]
             for m, p in zip(min_dists, pairs)]

    return pairs


def merge_pairs(pairs_a, pairs_b):
    res = pairs_a
    for pair in pairs_b:
        if [pair[1], pair[0]] not in res:
            res.append([pair[1], pair[0]])
    return res


def fill_gaps(pairs):
    diffs = []

    for p in pairs:
        try:
            diffs.append(p[0] - p[1])
        except TypeError:
            pass

    for i, p in enumerate(pairs):
        if p[0] is None:
            pairs[i][0] = pairs[i][1] + np.mean(diffs)
        if p[1] is None:
            pairs[i][1] = pairs[i][0] - np.mean(diffs)

    return pairs


def get_pairs(pts_a, pts_b):
    l_to_r = left_to_right(pts_a, pts_b)
    r_to_l = left_to_right(pts_b, pts_a)

    return fill_gaps(merge_pairs(l_to_r, r_to_l))

def get_line(pta, ptb):
    x1, y1 = pta
    x2, y2 = ptb
    k = (y2-y1)/(x2-x1)
    b = y1 - x1*k
    return k, b

def get_intersection(k1, b1, k2, b2):
    x = (b2-b1)/(k1-k2)
    y = k1*x + b1
    return x, y
