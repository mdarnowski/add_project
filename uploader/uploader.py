import json
import time

import pika
from loguru import logger
from pymongo import MongoClient

MONGO_URI = "mongodb://mongodb:27017/"
client = MongoClient(MONGO_URI)
db = client.bird_dataset
collection = db.images


def connect_to_rabbitmq() -> pika.BlockingConnection:
    """
    Connect to RabbitMQ server with retry on failure.
    """
    while True:
        try:
            connection = pika.BlockingConnection(pika.ConnectionParameters("rabbitmq"))
            logger.info("Connected to RabbitMQ.")
            return connection
        except pika.exceptions.AMQPConnectionError:
            logger.warning("RabbitMQ not available, retrying in 5 seconds...")
            time.sleep(5)


# RabbitMQ connection
connection = connect_to_rabbitmq()
channel = connection.channel()

channel.queue_declare(queue="processed_image_queue")


def callback(ch, method, properties, body) -> None:
    """
    Callback function to handle incoming messages from the RabbitMQ queue.
    """
    message = json.loads(body)
    try:
        # Extract and store only the necessary parts
        image_doc = {
            "image_path": message["image_path"],
            "data": message["data"],  # base64 encoded preprocessed image data
            "label": message["label"],
            "split": message["split"],
        }
        collection.insert_one(image_doc)
        logger.info(f"Inserted document with image_path: {message['image_path']}")
    except Exception as e:
        logger.error(f"Error inserting document: {e}")


channel.basic_consume(
    queue="processed_image_queue", on_message_callback=callback, auto_ack=True
)

logger.info("Uploader is listening for messages...")
channel.start_consuming()
