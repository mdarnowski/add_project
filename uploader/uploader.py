import base64
import json
import time

import gridfs
import pika
import pymongo
from bson import ObjectId
from loguru import logger
from pymongo import MongoClient

MONGO_URI = "mongodb://mongodb:27017/"
client = MongoClient(MONGO_URI)
db = client.bird_dataset
fs = gridfs.GridFS(db)
images_collection = db.images

# indexes for efficient querying
images_collection.create_index([("species", pymongo.ASCENDING)])
images_collection.create_index([("set_type", pymongo.ASCENDING)])
images_collection.create_index([("image_type", pymongo.ASCENDING)])


# Connect to RabbitMQ
def connect_to_rabbitmq() -> pika.BlockingConnection:
    while True:
        try:
            conn = pika.BlockingConnection(pika.ConnectionParameters("rabbitmq"))
            logger.info("Connected to RabbitMQ.")
            return conn
        except pika.exceptions.AMQPConnectionError:
            logger.warning("RabbitMQ not available, retrying in 5 seconds...")
            time.sleep(5)


connection = connect_to_rabbitmq()
channel = connection.channel()

channel.queue_declare(queue="raw_image_queue_uploader")
channel.queue_declare(queue="processed_image_queue")


# Save image to GridFS and store metadata in a single collection
def save_image_and_metadata(data: bytes, filename: str, metadata: dict) -> ObjectId:
    try:
        image_id = fs.put(data, filename=filename, metadata=metadata)
        images_collection.insert_one(
            {
                "filename": filename,
                "image_id": image_id,
                "image_type": metadata["image_type"],
                "species": metadata["species"],
                "set_type": metadata["set_type"],
            }
        )
        logger.info(f"Saved image {filename} with ID: {image_id}")
        return image_id
    except Exception as e:
        logger.error(f"Error saving image: {e}")
        return None


# Save raw image callback
def raw_callback(ch, method, properties, body) -> None:
    message = json.loads(body)
    raw_image_data = base64.b64decode(message["image_data"])
    metadata = {
        "image_type": "raw",
        "species": message["label"],
        "set_type": message["split"],
    }
    raw_image_id = save_image_and_metadata(
        raw_image_data, message["image_path"], metadata
    )


# Save processed image callback
def processed_callback(ch, method, properties, body) -> None:
    message = json.loads(body)
    processed_image_data = base64.b64decode(message["processed_image_data"])
    metadata = {
        "image_type": "processed",
        "species": message["label"],
        "set_type": message["split"],
    }
    processed_image_id = save_image_and_metadata(
        processed_image_data, message["image_path"], metadata
    )


channel.basic_consume(
    queue="raw_image_queue_uploader", on_message_callback=raw_callback, auto_ack=True
)
channel.basic_consume(
    queue="processed_image_queue", on_message_callback=processed_callback, auto_ack=True
)

logger.info("Uploader is listening for messages...")
channel.start_consuming()
