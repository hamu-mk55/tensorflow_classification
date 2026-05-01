import datetime

import numpy as np
import cv2

import tensorflow as tf
from tensorflow import keras

from .dataset import load_data_list, load_label_file
from .model import make_model
from .augmentation import image_augmentation


def make_dataset(file_list, img_w=224, img_h=224, img_ch=3, class_num=None):
    def split_info(info):
        ret = tf.strings.split(info, sep=" ")
        image_path = ret[0]
        label = ret[1]

        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=img_ch, expand_animations=False)
        image = tf.image.resize(image, [img_h, img_w])

        label = tf.strings.to_number(label, tf.int32)
        label = tf.one_hot(label, class_num)

        return image, label

    dataset = tf.data.Dataset.from_tensor_slices(file_list)

    dataset = dataset.map(split_info)

    return dataset


def augment_func(image, label):
    image_shape = image.shape
    [
        image,
    ] = tf.py_function(image_augmentation, [image], [tf.float32])

    image.set_shape(image_shape)

    return image, label


def show_dataset(dataset):
    for image, label in dataset.repeat():
        image = image.numpy().astype(np.uint8)
        cv2.imshow("test", image)
        cv2.waitKey(0)


class ImageClassification:
    def __init__(self):
        self._model = None

        self.label_list = None
        self.dataset_train = None
        self.dataset_test = None

        self.img_w = 224
        self.img_h = 224
        self.img_ch = 3
        self.batch_size = 16

    def load_label_data(self, label_file):
        self.label_list = load_label_file(label_file)

    def load_train_data(self, train_file, normalize=False):
        train_list = load_data_list(train_file)

        dataset_train = make_dataset(train_list, class_num=len(self.label_list))

        dataset_train = dataset_train.map(
            augment_func, num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset_train = dataset_train.cache().shuffle(
            len(train_list), reshuffle_each_iteration=True
        )
        dataset_train = dataset_train.batch(self.batch_size).prefetch(
            buffer_size=tf.data.AUTOTUNE
        )

        if normalize:
            normalization_layer = keras.layers.Rescaling(1.0 / 255)
            dataset_train = dataset_train.map(lambda x, y: (normalization_layer(x), y))

        self.dataset_train = dataset_train

    def load_test_data(self, test_file, normalize=False):
        test_list = load_data_list(test_file)

        dataset_test = make_dataset(test_list, class_num=len(self.label_list))
        dataset_test = dataset_test.cache()

        dataset_test = dataset_test.batch(self.batch_size).prefetch(
            buffer_size=tf.data.AUTOTUNE
        )

        if normalize:
            normalization_layer = keras.layers.Rescaling(1.0 / 255)
            dataset_test = dataset_test.map(lambda x, y: (normalization_layer(x), y))

        self.dataset_test = dataset_test

    def make_model(self):
        self._model = make_model(class_num=len(self.label_list))

    def load_model(self, model_path):
        self._model = keras.models.load_model(model_path)

    def train(
        self,
        label_file,
        train_file,
        test_file,
        epochs=100,
        premodel_path=None,
    ):
        self.load_label_data(label_file)
        self.load_train_data(train_file)
        self.load_test_data(test_file)

        if premodel_path is not None:
            self.load_model(premodel_path)
        else:
            self.make_model()

        loss_object = tf.keras.losses.CategoricalCrossentropy()
        train_accuracy = tf.keras.metrics.CategoricalAccuracy(name="train_accuracy")
        test_accuracy = tf.keras.metrics.CategoricalAccuracy(name="test_accuracy")

        optimizer = tf.keras.optimizers.Adam()
        train_loss = tf.keras.metrics.Mean(name="train_loss")
        test_loss = tf.keras.metrics.Mean(name="test_loss")

        @tf.function
        def train_step(images, labels):
            with tf.GradientTape() as tape:
                # training=True is only needed if there are layers with different
                # behavior during training versus inference (e.g. Dropout).
                predictions = self._model(images, training=True)
                loss = loss_object(labels, predictions)
            gradients = tape.gradient(loss, self._model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self._model.trainable_variables))

            train_loss(loss)
            train_accuracy(labels, predictions)

        @tf.function
        def test_step(images, labels):
            # training=False is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self._model(images, training=False)
            t_loss = loss_object(labels, predictions)

            test_loss(t_loss)
            test_accuracy(labels, predictions)

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = "logs/" + current_time + "/train"
        test_log_dir = "logs/" + current_time + "/test"
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)

        for epoch in range(epochs):
            # Reset the metrics at the start of the next epoch
            train_loss.reset_states()
            train_accuracy.reset_states()
            test_loss.reset_states()
            test_accuracy.reset_states()

            for images, labels in self.dataset_train:
                train_step(images, labels)

            with train_summary_writer.as_default():
                tf.summary.scalar("loss", train_loss.result(), step=epoch)
                tf.summary.scalar("accuracy", train_accuracy.result(), step=epoch)

            for test_images, test_labels in self.dataset_test:
                test_step(test_images, test_labels)

            with test_summary_writer.as_default():
                tf.summary.scalar("loss", test_loss.result(), step=epoch)
                tf.summary.scalar("accuracy", test_accuracy.result(), step=epoch)

            print(
                f"Epoch {epoch + 1}, "
                f"Loss: {train_loss.result():.3e}, "
                f"Accuracy: {train_accuracy.result() * 100:.2f}, "
                f"Test Loss: {test_loss.result():.3e}, "
                f"Test Accuracy: {test_accuracy.result() * 100:.2f}"
            )

        self._model.save_weights("./models/model.weights.h5")
        self._model.save("./models/model.keras")

    def evaluate(self, test_file, label_file=None):
        if label_file is not None:
            self.load_label_data(label_file)

        if self.label_list is None:
            raise ValueError(
                "label_list is not loaded. Call load_label_data() or pass label_file."
            )

        self.load_test_data(test_file)

        class_num = len(self.label_list)
        confusion_matrix = np.zeros((class_num, class_num), dtype=np.int32)

        for test_images, test_labels in self.dataset_test:

            preds = self._model.predict(test_images)
            true_labels = np.argmax(test_labels.numpy(), axis=1)
            pred_labels = np.argmax(preds, axis=1)

            for true_label, pred_label in zip(true_labels, pred_labels):
                confusion_matrix[true_label, pred_label] += 1

        total = np.sum(confusion_matrix)
        correct = np.trace(confusion_matrix)
        accuracy = correct / total if total > 0 else 0.0

        print(f"Accuracy: {accuracy * 100:.2f}% ({correct}/{total})")
        print("Class accuracy:")

        class_accuracy = {}
        for class_index, label_name in enumerate(self.label_list):
            class_total = np.sum(confusion_matrix[class_index])
            class_correct = confusion_matrix[class_index, class_index]
            accuracy_per_class = class_correct / class_total if class_total > 0 else 0.0
            class_accuracy[label_name] = accuracy_per_class
            print(
                f"  {label_name}: "
                f"{accuracy_per_class * 100:.2f}% ({class_correct}/{class_total})"
            )

        print("Confusion matrix:")
        print(confusion_matrix)

        return {
            "accuracy": accuracy,
            "class_accuracy": class_accuracy,
            "confusion_matrix": confusion_matrix,
        }

    def predict(self, image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(
            image, channels=self.img_ch, expand_animations=False
        )
        image = tf.image.resize(image, [self.img_h, self.img_w])

        # image = tf.cast(image, tf.float32) / 255.0
        image = tf.expand_dims(image, 0)
        return self._model.predict(image)
