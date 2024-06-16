import base64
import json
import os
import time
import gridfs
import pika
import pymongo
from loguru import logger
from pika.exceptions import AMQPConnectionError, ConnectionClosed
from pymongo import MongoClient, errors

MONGO_HOST = os.getenv("MONGO_HOST", "mongodb://localhost:27017/")
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
RAW_IMAGE_QUEUE = "raw_image_queue_uploader"
PROCESSED_IMAGE_QUEUE = "processed_image_queue"
TRAINING_METRICS_QUEUE = "training_metrics_queue"


class ImageUploader:
    def __init__(self) -> None:
        """
        Initializes the ImageUploader instance.

        Sets up the MongoDB and RabbitMQ connections, creates indexes, and declares queues.
        """
        self.client = MongoClient(MONGO_HOST)
        self.db = self.client.bird_dataset
        self.fs = gridfs.GridFS(self.db)
        self.images_collection = self.db.images
        self._create_indexes()
        self.connection = None
        self.channel = None
        self.metrics_collection = self.db.metrics
        self.raw_img_count = self.images_collection.count_documents(
            {"image_type": "raw"}
        )
        self.processed_img_count = self.images_collection.count_documents(
            {"image_type": "processed"}
        )
        self.connect_to_rabbitmq()

    def _create_indexes(self) -> None:
        self.images_collection.create_index([("species", pymongo.ASCENDING)])
        self.images_collection.create_index([("set_type", pymongo.ASCENDING)])
        self.images_collection.create_index([("image_type", pymongo.ASCENDING)])
        self.images_collection.create_index([("label", pymongo.ASCENDING)])
        self.images_collection.create_index(
            [("filename", pymongo.ASCENDING), ("image_type", pymongo.ASCENDING)],
            unique=True,
        )

    def connect_to_rabbitmq(self) -> None:
        while True:
            try:
                self.connection = pika.BlockingConnection(
                    pika.ConnectionParameters(RABBITMQ_HOST)
                )
                self.channel = self.connection.channel()
                self.channel.queue_declare(queue="TRAINING_METRICS_QUEUE")
                self.channel.queue_declare(queue=RAW_IMAGE_QUEUE)
                self.channel.queue_declare(queue=PROCESSED_IMAGE_QUEUE)
                logger.info("Connected to RabbitMQ.")
                break
            except AMQPConnectionError:
                logger.warning("RabbitMQ not available, retrying in 5 seconds...")
                time.sleep(5)

    def save_image_to_db(self, body: bytes, image_type: str) -> bool:
        try:
            message = json.loads(body)
            image_data = base64.b64decode(message["image_data"])
            filename = message["image_path"]
            image_id = self.fs.put(image_data, filename=filename)
            try:
                self.images_collection.insert_one(
                    {
                        "filename": filename,
                        "image_id": image_id,
                        "image_type": image_type,
                        "species": message["species"],
                        "set_type": message["split"],
                        "label": message["label"],
                    }
                )
                logger.info(f"Saved image {filename} with ID: {image_id}")
            except errors.DuplicateKeyError:
                logger.info(
                    f"Duplicate image ({filename}, {image_type}) detected in db."
                    f" Skipping save."
                )
                self.fs.delete(image_id)
                return False

            image_id = self.fs.put(image_data, filename=message["image_path"])
            self.images_collection.insert_one(
                {
                    "filename": message["image_path"],
                    "image_id": image_id,
                    "image_type": image_type,
                    "species": message["species"],
                    "set_type": message["split"],
                    "label": message["label"],
                }
            )
            logger.info(f"Saved image {message['image_path']} with ID: {image_id}")
            return True

        except Exception as e:
            logger.error(f"Error saving image: {e}")
            return False

    def process_and_save_metrics(self, body: bytes) -> None:
        try:
            message = json.loads(body)
            timestamp = message.pop("training_id")
            self.metrics_collection.update_one(
                {"timestamp": timestamp},
                {"$push": {"epochs": message}},
                upsert=True,
            )
            logger.info(f"Saved metrics for training ID: {timestamp}")
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")

    def handle_img(self, image_type: str, body: bytes) -> None:
        if self.save_image_to_db(body, image_type):
            if image_type == "raw":
                self.raw_img_count += 1
            elif image_type == "processed":
                self.processed_img_count += 1

        self.update_progress(image_type)

    def metrics_callback(self, _ch, _method, _properties, body) -> None:
        logger.info("Received metrics")
        self.process_and_save_metrics(body)

    def update_progress(self, image_type: str) -> None:
        """
        Sends progress update to the uploader progress queue based on image type.
        """
        if image_type == "raw":
            progress_message = {"uploaded": self.raw_img_count}
            self.channel.basic_publish(
                exchange="",
                routing_key="raw_uploader_progress_queue",
                body=json.dumps(progress_message),
            )
            logger.info(f"Sent raw upload progress update: {progress_message}")
        elif image_type == "processed":
            progress_message = {"uploaded": self.processed_img_count}
            self.channel.basic_publish(
                exchange="",
                routing_key="processed_uploader_progress_queue",
                body=json.dumps(progress_message),
            )
            logger.info(f"Sent processed upload progress update: {progress_message}")

    def start_consuming(self) -> None:
        while True:
            try:
                self.channel.basic_consume(
                    queue=RAW_IMAGE_QUEUE,
                    on_message_callback=lambda ch,
                    method,
                    properties,
                    body: self.handle_img("raw", body),
                    auto_ack=True,
                )
                self.channel.basic_consume(
                    queue=PROCESSED_IMAGE_QUEUE,
                    on_message_callback=lambda ch,
                    method,
                    properties,
                    body: self.handle_img("processed", body),
                    auto_ack=True,
                )
                self.channel.basic_consume(
                    queue="training_metrics_queue",
                    on_message_callback=self.metrics_callback,
                    auto_ack=True,
                )
                logger.info("Uploader is listening for messages...")
                self.channel.start_consuming()
            except (ConnectionClosed, AMQPConnectionError):
                logger.warning("Connection lost, reconnecting...")
                self.connect_to_rabbitmq()


if __name__ == "__main__":
    uploader = ImageUploader()
    uploader.start_consuming()
