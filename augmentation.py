import os
import random
import cv2
import numpy as np
import glob


def image_augmentation(img):
    img = img.numpy()

    img_shape = img.shape

    # 回転
    rand = random.random()
    if rand > 0.7:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif rand > 0.3:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # フリップ
    img = cv2.flip(img, random.randint(-1, 1))

    # zoom
    rand_top = random.randint(0, 5)
    rand_bottom = random.randint(0, 5)
    rand_right = random.randint(0, 5)
    rand_left = random.randint(0, 5)
    img = cv2.copyMakeBorder(
        img,
        rand_top,
        rand_bottom,
        rand_left,
        rand_right,
        cv2.BORDER_CONSTANT,
        (0, 0, 0),
    )
    img = cv2.resize(img, img_shape[0:2])

    return img


if __name__ == "__main__":
    pass
