import base64
import io
import json
import time
from typing import Dict, List, Tuple

import numpy as np
import pika
from PIL import Image
from pymongo import MongoClient
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Model

# MongoDB setup
MONGO_URI = "mongodb://mongodb:27017/"
client = MongoClient(MONGO_URI)
db = client.bird_dataset
weights_collection = db.model_weights  # Collection for model weights
collection = db.images  # Collection for image data


# Create a mapping from class names to integer labels
def create_class_mapping() -> Dict[str, int]:
    """
    Create a mapping from class names to integer labels.

    Returns:
        Dict[str, int]: Mapping of class names to integer labels.
    """
    labels = collection.distinct("label")
    class_mapping = {label: idx for idx, label in enumerate(sorted(labels))}
    return class_mapping


class_mapping = create_class_mapping()
print(f"Class Mapping: {class_mapping}")


# Model creation
def create_model(num_classes: int) -> Model:
    """
    Create a MobileNetV2-based model with custom dense layers.

    Args:
        num_classes (int): Number of output classes.

    Returns:
        tf.keras.models.Model: Compiled Keras model.
    """
    base_model = MobileNetV2(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3)
    )
    x = Flatten()(base_model.output)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model


# Instantiate and compile model
model = create_model(len(class_mapping))
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)


# Load dataset from MongoDB
def load_data(split: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load image data and labels from MongoDB.

    Args:
        split (str): Data split type ('train' or 'val').

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of image arrays and labels.
    """
    data_cursor = collection.find({"split": split})
    X: List[np.ndarray] = []
    y: List[int] = []
    for item in data_cursor:
        image_data = base64.b64decode(item["data"])
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        img_array = np.array(image) / 255.0

        label = item["label"]
        if label in class_mapping:
            X.append(img_array)
            y.append(class_mapping[label])
        else:
            print(f"Label '{label}' not found in class mapping. Skipping image.")

    return np.array(X), np.array(y)


# Training the model
def train_model(ch, method, properties, body) -> None:
    """
    Train the model with data from MongoDB.

    Args:
        ch: Channel.
        method: Method.
        properties: Properties.
        body: Message body containing training request.
    """
    message = json.loads(body)
    if message.get("train"):
        print("Starting model training...")

        X_train, y_train = load_data("train")
        X_val, y_val = load_data("val")

        if X_train.size == 0 or X_val.size == 0:
            print("No training or validation data found. Aborting training.")
            return

        model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

        # Save weights to MongoDB
        weights_id = weights_collection.insert_one(
            {"weights": [w.tolist() for w in model.get_weights()]}
        ).inserted_id
        channel.basic_publish(
            exchange="",
            routing_key="trained_model",
            body=json.dumps({"weights_id": str(weights_id)}),
        )
        print("Model training completed and weights sent.")


# Connect to RabbitMQ
def connect_to_rabbitmq() -> pika.BlockingConnection:
    """
    Connect to RabbitMQ server with retry on failure.

    Returns:
        pika.BlockingConnection: RabbitMQ connection object.
    """
    while True:
        try:
            connection = pika.BlockingConnection(pika.ConnectionParameters("rabbitmq"))
            print("Connected to RabbitMQ.")
            return connection
        except pika.exceptions.AMQPConnectionError:
            print("RabbitMQ not available, retrying in 5 seconds...")
            time.sleep(5)


# RabbitMQ setup for consuming
connection = connect_to_rabbitmq()
channel = connection.channel()

channel.queue_declare(queue="train_request")
channel.queue_declare(queue="trained_model")

channel.basic_consume(
    queue="train_request", on_message_callback=train_model, auto_ack=True
)

print("Trainer is waiting for training request...")
channel.start_consuming()
