import base64
import io
from typing import Tuple

import gridfs
import numpy as np
import tensorflow as tf
from PIL import Image
from pymongo import MongoClient
# Import ResNet50V2 and preprocessing function
from tensorflow.keras.applications.inception_v3 import (InceptionV3,
                                                        preprocess_input)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (BatchNormalization, Dense, Dropout,
                                     GlobalAveragePooling2D)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

# MongoDB setup
MONGO_URI = "mongodb://127.0.0.1:27017/"
client = MongoClient(MONGO_URI)
db = client.bird_dataset
collection = db.images
fs = gridfs.GridFS(db)


def create_model(num_classes: int) -> Model:
    # BEST RESNET101V2, RESNET50V2 but slow
    # Inceptionv3 fastesst and not bad
    # Propably need data augmention
    base_model = InceptionV3(include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    inputs = tf.keras.layers.Input(shape=(224, 224, 3), name="input-layer")
    x = base_model(inputs, training=False)  # Ensure batch norm layers are frozen
    x = GlobalAveragePooling2D(name="global_average_pooling_layer")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(
        num_classes,
        activation="softmax",
        kernel_regularizer=tf.keras.regularizers.l2(0.01),
        name="output-layer",
    )(x)
    model = Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.01
        ),  # Lower initial learning rate
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_data(split: str) -> Tuple[np.ndarray, np.ndarray, int]:
    data_cursor = collection.find({"set_type": split, "image_type": "processed"})
    X, y = [], []
    label_to_index = {}
    current_index = 0

    for item in data_cursor:
        image_data = fs.get(item["image_id"]).read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        img_array = np.array(image)
        img_array = preprocess_input(img_array)

        label = item["label"]
        if label not in label_to_index:
            label_to_index[label] = current_index
            current_index += 1

        X.append(img_array)
        y.append(label_to_index[label])

    X = np.array(X)
    y = to_categorical(np.array(y), num_classes=current_index)
    return X, y, current_index


def train_and_evaluate_model() -> None:
    print("Starting model training...")

    X_train, y_train, num_classes = load_data("train")
    X_val, y_val, _ = load_data("val")
    X_test, y_test, _ = load_data("test")

    model = create_model(num_classes)

    # Define callbacks
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=2, min_lr=0.00001
    )

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    print("Training with frozen base model...")
    model.fit(
        X_train,
        y_train,
        epochs=10,
        validation_data=(X_val, y_val),
        callbacks=[reduce_lr, early_stopping],
    )

    print("Fine-tuning the model...")
    base_model = model.layers[1]  # Assuming base model is the second layer
    base_model.trainable = True
    for layer in base_model.layers[:-10]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        X_train,
        y_train,
        epochs=10,
        validation_data=(X_val, y_val),
        callbacks=[reduce_lr, early_stopping],
    )

    print("Evaluating on test data...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")


if __name__ == "__main__":
    train_and_evaluate_model()
