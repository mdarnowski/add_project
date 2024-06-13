import gc
import io
from typing import Tuple
import gridfs
import keras.backend
import numpy as np
import tensorflow as tf
from PIL import Image
from pymongo import MongoClient
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
)
from tensorflow.keras.models import Model
from tensorflow.python.keras.utils.np_utils import to_categorical

# MongoDB setup
MONGO_URI = "mongodb://127.0.0.1:27017/"
client = MongoClient(MONGO_URI)
db = client.bird_dataset
collection = db.images
fs = gridfs.GridFS(db)


def create_model(num_classes: int) -> Model:
    # All with 70 bird types
    # BEST RESNET101V2(64% acc, RESNET50V2(55% acc) but slow
    # Inception3(55% acc) fastest and not bad
    # Probably need data augmentation
    base_model = InceptionV3(include_top=False, input_shape=(299, 299, 3))
    base_model.trainable = False

    inputs = tf.keras.layers.Input(shape=(299, 299, 3), name="input-layer")
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
            learning_rate=0.001
        ),  # Lower initial learning rate
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_data(split: str) -> Tuple[np.ndarray, np.ndarray, int]:
    data_cursor = collection.find({"set_type": split, "image_type": "processed"})
    X, y = [], []

    for item in data_cursor:
        # Read the image
        image_data = fs.get(item["image_id"]).read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        img_array = np.array(image) / 255.0

        # Get the label directly
        label = item["label"]

        X.append(img_array)
        y.append(label)
    num_classes = len(np.unique(y))
    X = np.array(X)
    y = to_categorical(np.array(y), num_classes=num_classes)

    # Assuming you need the number of unique classes

    return X, y, num_classes


def train_and_evaluate_model() -> None:
    print("Starting model training...")

    X_train, y_train, num_classes = load_data("train")
    print("Train data loaded")
    X_val, y_val, _ = load_data("val")
    print("Val data loaded")
    print(f"Number of classes: {num_classes}")
    model = create_model(num_classes)
    print("X_train shape: ", X_train.shape)
    print("y_train shape: ", y_train.shape)
    print("X_val shape: ", X_val.shape)
    print("y_val shape: ", y_val.shape)
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
        batch_size=16,
        epochs=2,
        validation_data=(X_val, y_val),
        callbacks=[reduce_lr, early_stopping],
    )

    print("Fine-tuning the model...")
    base_model = model.layers[1]  # Assuming base model is the second layer
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=16,
        validation_data=(X_val, y_val),
        callbacks=[reduce_lr, early_stopping],
    )
    del X_train, y_train, X_val, y_val
    gc.collect()
    keras.backend.clear_session()
    print("X_train, y_train and X_val, y_val deleted. Loading test data...")
    X_test, y_test, _ = load_data("test")
    print("Test data loaded")
    print("Evaluating on test data...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
    print("Saving model...")
    model.save("model.h5")


if __name__ == "__main__":
    train_and_evaluate_model()
