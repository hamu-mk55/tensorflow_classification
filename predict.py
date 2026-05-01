import os
import glob

from src.classification import ImageClassification


def predict():
    # InfoとWarningを非表示
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    model = ImageClassification()
    model.load_model("./models/model.keras")

    image_paths = glob.glob("./data/images/*/*.png")
    for image_path in image_paths:
        pred = model.predict(image_path)
        print(image_path, pred)


if __name__ == "__main__":
    predict()
