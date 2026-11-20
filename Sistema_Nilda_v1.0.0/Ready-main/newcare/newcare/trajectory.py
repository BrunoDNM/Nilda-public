"""Independent, minimal implementation of the Nilda trajectory representation.

Computations implement the numerical protocol recorded in the paper. This file
is newly written for the research release and does not ship SmartCare's code.
"""
import json
import numpy as np
from skimage.transform import resize
from .Config import Config


def load_json_data(filename):
    with open(filename, encoding='utf-8') as f:
        data = json.load(f)
    # The source trajectory generator stored a JSON string containing JSON.
    return json.loads(data) if isinstance(data, str) else data


def tortuosity(points):
    deltas = np.diff(points, axis=0)
    length = np.linalg.norm(deltas, axis=1).sum()
    chord = np.linalg.norm(points[-1] - points[0])
    return float(length / chord) if chord >= 1e-8 else 1.0


def fractal_dimension(points):
    box_sizes = np.array([2, 4, 8, 16, 32, 64])
    x, y = points[:, 0], points[:, 1]
    minimum = points.min(axis=0)
    span = points.max(axis=0) - minimum
    occupied = []
    for size in box_sizes:
        dims = np.maximum(np.ceil(span / size).astype(int), 1)
        cell = np.minimum(np.maximum(((points - minimum) / size).astype(int), 0), dims - 1)
        # Same box-counting rule, implemented using unique cells rather than a grid.
        occupied.append(len(np.unique(cell, axis=0)))
    counts = np.asarray(occupied, dtype=float)
    if sum(counts > 0) < 2:
        return 1.0
    return float(np.polyfit(np.log(1.0 / box_sizes), np.log(counts), 1)[0])


def back_and_forth_count(points):
    deltas = np.diff(points, axis=0)
    total = 0
    for coord in range(2):
        directions = np.sign(deltas[:, coord])
        directions = directions[directions != 0]
        total += int(np.count_nonzero(directions[1:] != directions[:-1]))
    return float(total)


def room_transitions(points, x_mid=300, y_mid=150):
    rooms = (points[:, 0] >= x_mid).astype(int) + 2 * (points[:, 1] >= y_mid).astype(int)
    return float(np.count_nonzero(rooms[1:] != rooms[:-1]))


def extract_features_from_trajectory(points, x_mid=300, y_mid=150):
    return np.array((tortuosity(points), fractal_dimension(points),
                     back_and_forth_count(points),room_transitions(points,x_mid,y_mid)))


def _cells_on_segment(x0, y0, x1, y1):
    """Sample both neighbouring pixels along the segment's dominant axis."""
    if abs(y1-y0) > abs(x1-x0):
        # Run along y and invert axes back to row/column coordinates.
        cols, rows = _cells_on_segment(y0, x0, y1, x1)
        return rows, cols
    if x1 < x0:
        return _cells_on_segment(x1, y1, x0, y0)
    if x0 == x1:
        return np.asarray([y0, y1],dtype=int), np.asarray([x0,x1],dtype=int)
    xs = np.arange(x0, x1+1,dtype=float)
    ys = xs*(y1-y0)/(x1-x0) + (x1*y0-x0*y1)/(x1-x0)
    floors=np.floor(ys).astype(int)
    return np.concatenate((floors,floors+1)), np.concatenate((xs.astype(int), xs.astype(int)))


def render_trajectory(points, img_height=602, img_width=369, target_size=None):
    picture=np.zeros((img_height,img_width))
    last=None
    for x0,y0 in points:
        x,y=int(x0),int(y0)
        if 0 <= y < img_height and 0 <= x < img_width:
            picture[y,x]=1
        if last is not None:
            rows,cols=_cells_on_segment(last[0],last[1],x,y)
            picture[np.clip(rows,0,img_height-1),np.clip(cols,0,img_width-1)]=1
        last=(x,y)
    if target_size is not None:
        small=resize(picture,(target_size,target_size))
        return small.reshape((1,target_size,target_size,1)).astype(np.float32)
    return picture
