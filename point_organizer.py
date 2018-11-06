import statistics as stat
import numpy as np


# каждой точке из left_pts ищет пару из right_pts
def left_to_right(left_pts, right_pts):
    if len(left_pts) > len(right_pts):
        left_pts, right_pts = right_pts, left_pts

    pairs = []

    for lp in left_pts:
        dists = [abs(lp - rp) for rp in right_pts]
        pairs.append([lp, right_pts.pop(dists.index(min(dists)))])

    for rp in right_pts:
        pairs.append([None, rp])

    return pairs


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
    return fill_gaps(left_to_right(pts_a, pts_b))


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
