import json
import time

import numpy as np
import pika
import tensorflow as tf
from pymongo import MongoClient
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Model

# MongoDB setup
MONGO_URI = "mongodb://mongodb:27017/"
client = MongoClient(MONGO_URI)
db = client.bird_dataset
weights_collection = db.model_weights  # Collection for model weights


# Model creation
def create_model() -> Model:
    """
    Create a MobileNetV2-based model with custom dense layers.

    Returns:
        tf.keras.models.Model: Compiled Keras model.
    """
    base_model = MobileNetV2(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3)
    )
    x = Flatten()(base_model.output)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    predictions = Dense(200, activation="softmax")(x)  # Assuming 200 classes

    model = Model(inputs=base_model.input, outputs=predictions)
    return model


# Instantiate model
model = create_model()


# Load model weights from MongoDB
def load_weights(weights_id: str) -> None:
    """
    Load model weights from MongoDB by weights_id.

    Args:
        weights_id (str): The identifier for the weights document in MongoDB.
    """
    weights_doc = weights_collection.find_one({"_id": weights_id})
    if weights_doc:
        weights = [np.array(w) for w in weights_doc["weights"]]
        model.set_weights(weights)
        print("Model weights loaded successfully.")
    else:
        print("Weights not found in MongoDB.")


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


# Establish RabbitMQ connection and channel
connection = connect_to_rabbitmq()
channel = connection.channel()
channel.queue_declare(queue="trained_model")


# Message handler
def on_weights_received(ch, method, properties, body) -> None:
    """
    Callback function to handle incoming weight messages.

    Args:
        ch: Channel.
        method: Method.
        properties: Properties.
        body: Message body containing weights_id.
    """
    message = json.loads(body)
    weights_id = message["weights_id"]
    load_weights(weights_id)


# Start consuming messages
channel.basic_consume(
    queue="trained_model", on_message_callback=on_weights_received, auto_ack=True
)

print("Predictor is waiting for model weights...")
channel.start_consuming()
