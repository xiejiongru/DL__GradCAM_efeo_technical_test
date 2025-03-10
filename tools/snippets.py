import pickle
import logging
from pathlib import Path

import cv2


def mkdir(directory) -> Path:
    """
    Python 3.5 pathlib shortcut to mkdir -p
    Fails if parent is created by other process in the middle of the call
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


class Averager(object):
    """
    Taken from kensh code. Also seen in Gunnar's code
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.last = 0.0
        self.avg = 0.0
        self._sum = 0.0
        self._count = 0.0

    def update(self, value, weight=1):
        self.last = value
        self._sum += value * weight
        self._count += weight
        self.avg = self._sum / self._count

    def __repr__(self):
        return 'Averager[{:.4f} (A: {:.4f})]'.format(self.last, self.avg)


def visualize_bbox(
        img, bbox, class_name,
        BOX_COLOR=(255, 0, 0),
        TEXT_COLOR=(255, 255, 255),
        thickness=2):
    """Visualizes a single bounding box on the image"""
    x_min, y_min, x_max, y_max = map(int, bbox)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max),
            color=BOX_COLOR, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(
            class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)

    cv2.rectangle(img, (x_min, y_max - int(1.3 * text_height)),
            (x_min + text_width, y_max), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_max - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    return img


def quick_log_setup(level):
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
            "%(asctime)s %(name)s %(levelname)s: %(message)s",
            "%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def compute_or_load_pkl_silently(filepath, function, *args, **kwargs):
    """Implementation without outputs"""
    try:
        with Path(filepath).open('rb') as f:
            pkl = pickle.load(f)
    except (EOFError, FileNotFoundError):
        pkl = function(*args, **kwargs)
        with Path(filepath).open('wb') as f:
            pickle.dump(pkl, f, pickle.HIGHEST_PROTOCOL)
    return pkl


def stash2(stash_to):
    def stash_func(function, *args, **kwargs):
        return compute_or_load_pkl_silently(stash_to, function, *args, **kwargs)
    return stash_func
