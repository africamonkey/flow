import numpy as np


EPS = 1e-6


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Segment:
    def __init__(self, a, b):
        self.a = a
        self.b = b


def cross_product(base, a, b):
    """cross product. Point base, Point a, Point b"""
    x1, y1, x2, y2 = a.x - base.x, a.y - base.y, b.x - base.x, b.y - base.y
    return x1 * y2 - x2 * y1


def sgn(x):
    if x > EPS:
        return 1
    if x < -EPS:
        return -1
    return 0


def cmp(x, y):
    return sgn(x - y)


def is_segment_intersect(l1, l2):
    s1, e1, s2, e2 = l1.a, l1.b, l2.a, l2.b
    if (cmp(min(s1.x, e1.x), max(s2.x, e2.x)) <= 0) and \
       (cmp(min(s1.y, e1.y), max(s2.y, e2.y)) <= 0) and \
       (cmp(min(s2.x, e2.x), max(s1.x, e1.x)) <= 0) and \
       (cmp(min(s2.y, e2.y), max(s1.y, e1.y)) <= 0) and \
       (sgn(cross_product(s1, s2, e2)) * sgn(cross_product(e1, s2, e2)) <= 0) and \
       (sgn(cross_product(s2, s1, e1)) * sgn(cross_product(e2, s1, e1)) <= 0):
        return True
    return False


def intersect_point(a, b):
    s1 = cross_product(b.a, a.a, b.b)
    s2 = cross_product(b.a, a.b, b.b)
    try:
        ret = Point((a.b.x * s1 - a.a.x * s2) / (s1 - s2), (a.b.y * s1 - a.a.y * s2) / (s1 - s2))
    except ZeroDivisionError:
        return None
    return ret


def det(a, b):
    x1, y1, x2, y2 = a[0], a[1], b[0], b[1]
    return x1 * y2 - x2 * y1


def dis(a, b):
    x1, y1, x2, y2 = a[0], a[1], b[0], b[1]
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def radius_convert(x):
    return (450 - x) / 180 * np.pi


def car_collide_point(ori_a, ori_b):
    time_eval = 1000
    angle_a = radius_convert(ori_a[2])
    x1, y1 = ori_a[0], ori_a[1]
    x2, y2 = x1 + np.cos(angle_a) * time_eval, y1 + np.sin(angle_a) * time_eval
    sega = Segment(Point(x1, y1), Point(x2, y2))
    angle_b = radius_convert(ori_b[2])
    x1, y1 = ori_b[0], ori_b[1]
    x2, y2 = x1 + np.cos(angle_b) * time_eval, y1 + np.sin(angle_b) * time_eval
    segb = Segment(Point(x1, y1), Point(x2, y2))
    if is_segment_intersect(sega, segb) is False:
        return None
    return intersect_point(sega, segb)


def car_ttc(ori_a, ori_b, v_a, v_b):
    col_point = car_collide_point(ori_a, ori_b)
    if col_point is None or abs(v_b) < 1e-3:
        return 1000
    col_ori = [col_point.x, col_point.y]
    return dis(ori_b, col_ori) / abs(v_b)
