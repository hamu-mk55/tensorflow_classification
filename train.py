import os

from src.dataset import make_data_list
from src.classification import ImageClassification


def train():
    # InfoとWarningを非表示
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    make_data_list(
        src_dir="./data/images",
        out_dir="./data",
        file_ext="png",
        ratio=0.8,
        shuffle=True,
    )

    model = ImageClassification()
    model.train(
        train_file="./data/train.txt",
        label_file="./data/labels.txt",
        test_file="./data/test.txt",
        epochs=1,
        premodel_path="./models/model.keras",
    )

    model.evaluate(test_file="./data/test.txt")


if __name__ == "__main__":
    train()
