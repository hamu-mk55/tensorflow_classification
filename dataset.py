import os
import random
import cv2
import numpy as np
import glob


def make_data_list(
    file_ext=".jpg",
    src_dir="./images",
    out_dir="./data",
    ratio=0.75,
    shuffle=True,
    mode="both",
):

    os.makedirs(out_dir, exist_ok=True)

    labels = glob.glob(f"{src_dir}/*")

    labelsTxt = open(f"{out_dir}/labels.txt", "w")

    if mode != "train":
        test = open(f"{out_dir}/test.txt", "w")
    if mode != "test":
        train = open(f"{out_dir}/train.txt", "w")

    classNo = 0
    cnt = 0
    for label in labels:
        if not os.path.isdir(label):
            continue

        label = os.path.relpath(label, start=src_dir)

        labelsTxt.write(label + "\n")

        work_dir = f"{src_dir}/{label}"
        image_files = glob.glob(f"{work_dir}/*.{file_ext}")

        if shuffle:
            random.shuffle(image_files)

        start_cnt = cnt
        length = len(image_files)
        for image_file in image_files:

            if cnt - start_cnt < length * ratio:
                if mode != "test":
                    train.write(f"{image_file} {classNo}\n")
            else:
                if mode != "train":
                    test.write(f"{image_file} {classNo}\n")

            cnt += 1

        print("classNo: ", classNo, "label: ", label, "num: ", cnt - start_cnt)
        classNo += 1

    if mode != "test":
        train.close()
    if mode != "train":
        test.close()
    labelsTxt.close()

    return 1


def load_label_file(label_file):
    label_list = []

    with open(label_file) as labelfile:
        for line in labelfile:
            label_list.append(line.strip())

    return label_list


def load_data_list(data_file):
    data_list = []

    with open(data_file) as datafile:
        for line in datafile:
            data_list.append(line)

    return data_list


if __name__ == "__main__":
    make_data_list(file_ext="png")
