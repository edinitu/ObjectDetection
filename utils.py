import numpy as np


def grey2rgb(img):
    assert img.shape == (448, 448)
    new_img = np.zeros((448, 448, 3))
    new_img[:, :, 0] = img
    new_img[:, :, 1] = img
    new_img[:, :, 2] = img
    return new_img
