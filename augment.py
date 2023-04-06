from torch import Tensor, from_numpy
import torchvision.transforms.functional as f
from random import random
import numpy as np
import cv2


def randfloat(l:float, r:float) -> float:
    if r == l: return l
    return l + random() * (r - l)


def affine(*samples: Tensor, trans_r=[-0.5, 0.5], scale_r=[0.5, 1.5], shear=[0.5, 1]):
    # -180 ~ +180 degrees
    angle = randfloat(-180, 180)
    # Depending on size of the tensor
    tH, tW = [randfloat(*trans_r) for _ in range(2)]
    def translate(img: Tensor):
        H, W = list(img.shape)[-2:]
        return (int(H * tH), int(W * tW))
    # Any float number
    scale = randfloat(*scale_r)
    # Any float number
    shear = [randfloat(*shear) for _ in range(2)]
    # mapping callback
    def apply(sample: Tensor):
        return f.affine(
            sample,
            angle=angle,
            translate=translate(sample),
            scale=scale,
            shear=shear
        )
    # perform transform on all given samples
    return list(map(apply, samples))

if __name__ == "__main__":
    x, y = np.mgrid[:900, :900]
    img = np.logical_xor((x % 100) < 50, (y % 100) < 50)
    img = np.stack([img, ~img, np.zeros(img.shape)], axis=2).astype(np.float32)
    cv2.imshow("original", img)
    img = from_numpy(img)
    while True:
        a, = affine(img).detach().numpy().astype(np.float32)
        cv2.imshow("affine", img)
        k = cv2.waitKey(0)
        if k != ' ' or k != 13: break
