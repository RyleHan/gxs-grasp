"""Rotated grasp rectangles: drawing training maps, decoding, and the IoU metric.

Rectangles are (x, y, w, h, theta_deg) in pixels, as stored in Grasp-Anything++:
(x, y) centre, w = gripper opening, h = jaw size, theta in degrees. Geometry and
the success criterion follow the GR-ConvNet / LGD code (MIT licensed) so that our
numbers use the same protocol: IoU > 0.25 and angle difference < 30 degrees.
"""
import numpy as np
from skimage.draw import polygon

SRC_SIZE = 416          # Grasp-Anything images
W_NORM = 112.0          # width/height maps are normalised by output_size / 2 (224 / 2)


def corners(rect):
    """Four corners as (row, col); same construction as GR-ConvNet's Grasp.as_gr."""
    x, y, w, h, th = rect
    a = -np.deg2rad(th)
    xo, yo = np.cos(a), np.sin(a)
    y1, x1 = y + w / 2 * yo, x - w / 2 * xo
    y2, x2 = y - w / 2 * yo, x + w / 2 * xo
    return np.array([[y1 - h / 2 * xo, x1 - h / 2 * yo],
                     [y2 - h / 2 * xo, x2 - h / 2 * yo],
                     [y2 + h / 2 * xo, x2 + h / 2 * yo],
                     [y1 + h / 2 * xo, x1 + h / 2 * yo]])


def scale(rects, s):
    r = np.array(rects, dtype=np.float32).reshape(-1, 5).copy()
    r[:, :4] *= s
    return r


def draw_maps(rects, size):
    """GR-ConvNet style targets from rects already in output pixels.

    Quality is 1 on the centre third of each rectangle (along the opening axis);
    angle is encoded as (cos 2a, sin 2a) because a parallel gripper is symmetric
    under 180 degrees; w and h are normalised by W_NORM.
    Returns float32 array (5, size, size): pos, cos, sin, w, h.
    """
    out = np.zeros((5, size, size), dtype=np.float32)
    for x, y, w, h, th in rects:
        rr, cc = polygon(*corners((x, y, w / 3, h, th)).T, shape=(size, size))
        a = -np.deg2rad(th)
        out[0, rr, cc] = 1.0
        out[1, rr, cc] = np.cos(2 * a)
        out[2, rr, cc] = np.sin(2 * a)
        out[3, rr, cc] = min(w, W_NORM) / W_NORM
        out[4, rr, cc] = min(h, W_NORM) / W_NORM
    return out


def region_mask(rects_src, grid=28, cell=16):
    """Union of full rectangles, pooled to the CLIP patch grid (bool, grid x grid).

    rects_src are in 416-px source coordinates; we rasterise at grid*cell (448,
    the CLIP input size) and mark a patch if any pixel of it is covered.
    """
    size = grid * cell
    canvas = np.zeros((size, size), dtype=bool)
    for r in scale(rects_src, size / SRC_SIZE):
        rr, cc = polygon(*corners(r).T, shape=(size, size))
        canvas[rr, cc] = True
    return canvas.reshape(grid, cell, grid, cell).any(axis=(1, 3))


def decode(q, cos, sin, w, h):
    """Best grasp from dense maps: argmax of q, then read geometry at that pixel."""
    r, c = np.unravel_index(np.argmax(q), q.shape)
    a = 0.5 * np.arctan2(sin[r, c], cos[r, c])
    return np.array([c, r, max(w[r, c], 0) * W_NORM, max(h[r, c], 0) * W_NORM, -np.rad2deg(a)],
                    dtype=np.float32)


def iou(ra, rb, angle_thr=30.0):
    """Rasterised IoU of two rectangles, 0 if their angles differ by > angle_thr (mod 180)."""
    d = abs((ra[4] - rb[4] + 90.0) % 180.0 - 90.0)
    if d > angle_thr:
        return 0.0
    pa, pb = corners(ra), corners(rb)
    hi = np.ceil(np.maximum(pa.max(0), pb.max(0))).astype(int) + 1
    lo = np.floor(np.minimum(pa.min(0), pb.min(0))).astype(int)
    shape = tuple(hi - lo)
    ma = np.zeros(shape, dtype=bool)
    mb = np.zeros(shape, dtype=bool)
    ma[polygon(*(pa - lo).T, shape=shape)] = True
    mb[polygon(*(pb - lo).T, shape=shape)] = True
    union = (ma | mb).sum()
    return float((ma & mb).sum() / union) if union else 0.0


def success(pred, gts, thr=0.25):
    return any(iou(pred, g) > thr for g in gts)
