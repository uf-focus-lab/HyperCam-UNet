from torch import Tensor, from_numpy, flip, cat, mean
import torchvision.transforms.functional as f
from random import random
import numpy as np
import cv2


def randfloat(l: float, r: float) -> float:
    if r == l:
        return l
    return l + random() * (r - l)


def affine(
    *samples: Tensor, rot_r=[-30, 30], trans_r=[-0.2, 0.2], scale_r=[0.8, 1.6], shear=[-5, 5]
):
    # -180 ~ +180 degrees
    angle = randfloat(*rot_r)
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
        trans = translate(sample)
        fill = [float(_) for _ in mean(sample, dim=[0, 2, 3]).detach().cpu()]
        # b, c, h, w = sample.shape
        # Pad the sample with the mirror of itself to avoid black borders
        # flip_h = flip(sample, [3])  # Flip along the width dimension
        # Pad along width
        # sample = cat((flip_h, sample, flip_h), dim=3)
        # flip_v = flip(sample, [2])  # Flip along the height dimension
        # Pad along height
        # sample = cat((flip_v, sample, flip_v), dim=2)
        # Apply affine transformation
        sample = (
            f.affine(
                sample, angle=angle, translate=trans, scale=scale, shear=shear, fill=fill
            )
        )
        # Crop the sample to original size
        # sample = sample[:, :, h : 2 * h, w : 2 * w].contiguous()
        return sample

    # perform transform on all given samples
    return list(map(apply, samples))


if __name__ == "__main__":
    x, y = np.mgrid[:250, :250]
    checker = np.logical_xor((x % 100) < 50, (y % 100) < 50)
    checker = np.stack([checker, ~checker, np.zeros(checker.shape)], axis=2).astype(
        np.float32
    )
    img = np.zeros((550, 550, 3), checker.dtype)
    img[150:-150, 150:-150] = checker
    cv2.imshow("original", img)
    img = from_numpy(img).swapaxes(0, 2)
    while True:
        (a,) = affine(img)
        a = a.detach().swapaxes(0, 2).numpy().astype(np.float32)
        cv2.imshow("affine", a)
        k = cv2.waitKey(0)
        if k == ord("q"):
            break
