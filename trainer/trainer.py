import base64
import io
import json
import time
from typing import Dict, List, Tuple

import gridfs
import matplotlib.pyplot as plt
import numpy as np
import pika
from PIL import Image
from pymongo import MongoClient
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.python.keras.callbacks import EarlyStopping

# MongoDB setup
MONGO_URI = "mongodb://127.0.0.1:27017/"
client = MongoClient(MONGO_URI)
db = client.bird_dataset
weights_collection = db.model_weights
collection = db.images
fs = gridfs.GridFS(db)


def create_model(num_classes: int) -> Model:
    """Create a MobileNetV2-based model with custom dense layers."""
    base_model = MobileNetV2(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3)
    )
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def load_data(
    split: str, visualize: bool = False
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Load image data and labels from MongoDB."""
    data_cursor = collection.find({"set_type": split, "image_type": "processed"})
    X, y = [], []
    images_to_visualize, labels_to_visualize = [], []

    label_to_index = {}
    current_index = 0

    for item in data_cursor:
        image_data = fs.get(item["image_id"]).read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        img_array = np.array(image) / 255.0

        label = item["label"]
        if label not in label_to_index:
            label_to_index[label] = current_index
            current_index += 1

        X.append(img_array)
        y.append(label_to_index[label])

        if visualize and len(images_to_visualize) < 5:
            images_to_visualize.append(image)
            labels_to_visualize.append(label)

    if visualize:
        for img, lbl in zip(images_to_visualize, labels_to_visualize):
            plt.figure()
            plt.imshow(img)
            plt.title(f"{split.capitalize()} Image - Label: {lbl}")
            plt.axis("off")
            plt.show()

    return np.array(X), np.array(y), len(label_to_index)


def train_and_evaluate_model() -> None:
    """Train the model with data from MongoDB and evaluate on test data."""
    print("Starting model training...")

    X_train, y_train, num_classes_train = load_data("train", visualize=True)
    X_val, y_val, num_classes_val = load_data("val", visualize=True)
    X_test, y_test, num_classes_test = load_data("test", visualize=True)

    if X_train.size == 0 or X_val.size == 0 or X_test.size == 0:
        print("No training, validation, or test data found. Aborting training.")
        return

    num_classes = max(num_classes_train, num_classes_val, num_classes_test)
    model = create_model(num_classes)

    model.fit(X_train, y_train, epochs=40, validation_data=(X_val, y_val))
    print("Model training completed. Starting evaluation on test data...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")


def connect_to_rabbitmq() -> pika.BlockingConnection:
    """Connect to RabbitMQ server with retry on failure."""
    while True:
        try:
            connection = pika.BlockingConnection(pika.ConnectionParameters("rabbitmq"))
            print("Connected to RabbitMQ.")
            return connection
        except pika.exceptions.AMQPConnectionError:
            print("RabbitMQ not available, retrying in 5 seconds...")
            time.sleep(5)


if __name__ == "__main__":
    train_and_evaluate_model()
