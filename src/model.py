from tensorflow import keras


class CNNModel(keras.Model):
    def __init__(self, class_num):
        super().__init__()
        self.conv1 = keras.layers.Conv2D(16, 3, activation="relu")
        self.conv2 = keras.layers.Conv2D(32, 3, activation="relu")
        self.conv3 = keras.layers.Conv2D(64, 3, activation="relu")

        self.pool = keras.layers.MaxPooling2D()
        self.flatten = keras.layers.Flatten()

        self.d1 = keras.layers.Dense(128, activation="relu")
        self.d2 = keras.layers.Dense(class_num)

    def call(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)

        x = self.flatten(x)
        x = self.d1(x)

        return self.d2(x)


def make_model(class_num):
    case = 2

    # Case-0: from scratch
    if case == 1:
        model = keras.applications.efficientnet_v2.EfficientNetV2B0(
            include_top=True, weights=None, classes=class_num, input_shape=(224, 224, 3)
        )

    # Case-1: pre-trained weights
    elif case == 2:
        inputs = keras.layers.Input(shape=(224, 224, 3))
        model = keras.applications.efficientnet_v2.EfficientNetV2B0(
            include_top=False, input_tensor=inputs, weights="imagenet"
        )

        # Freeze the pretrained weights
        model.trainable = False

        for layer in model.layers[-33:]:
            print(layer.name)
            if not isinstance(layer, keras.layers.BatchNormalization):
                layer.trainable = True

        # Rebuild top
        x = keras.layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
        x = keras.layers.BatchNormalization()(x)

        top_dropout_rate = 0.2
        x = keras.layers.Dropout(top_dropout_rate, name="top_dropout")(x)
        outputs = keras.layers.Dense(class_num, activation="softmax", name="pred")(x)
        model = keras.Model(inputs, outputs, name="EfficientNet")

    # Case: Custom Model
    else:
        model = CNNModel(class_num=class_num)
        model.build((None, 224, 224, 3))

    print(model.summary())

    return model


if __name__ == "__main__":
    make_model(class_num=6)
