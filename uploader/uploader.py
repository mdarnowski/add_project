import base64
import json
import os
import time
import gridfs
import pika
import pymongo
from loguru import logger
from pika.exceptions import AMQPConnectionError, ConnectionClosed
from pymongo import MongoClient


class ImageUploader:
    def __init__(self) -> None:
        self.mongo_uri = os.getenv("MONGO_HOST", "mongodb://localhost:27017/")
        self.rabbitmq_host = os.getenv("RABBITMQ_HOST", "localhost")
        self.client = MongoClient(self.mongo_uri)
        self.db = self.client.bird_dataset
        self.fs = gridfs.GridFS(self.db)
        self.images_collection = self.db.images
        self._create_indexes()
        self.connection = None
        self.channel = None
        self.metrics_collection = self.db.metrics
        self.total_raw_images = 0
        self.uploaded_raw_images = 0
        self.total_processed_images = 0
        self.uploaded_processed_images = 0
        self.connect_to_rabbitmq()

    def _create_indexes(self) -> None:
        self.images_collection.create_index([("species", pymongo.ASCENDING)])
        self.images_collection.create_index([("set_type", pymongo.ASCENDING)])
        self.images_collection.create_index([("image_type", pymongo.ASCENDING)])
        self.images_collection.create_index([("label", pymongo.ASCENDING)])

    def connect_to_rabbitmq(self) -> None:
        while True:
            try:
                self.connection = pika.BlockingConnection(
                    pika.ConnectionParameters(self.rabbitmq_host)
                )
                self.channel = self.connection.channel()
                self.channel.queue_declare(queue="raw_image_queue_uploader")
                self.channel.queue_declare(queue="processed_image_queue")
                self.channel.queue_declare(queue="training_metrics_queue")
                self.channel.queue_declare(queue="raw_uploader_progress_queue")
                self.channel.queue_declare(queue="processed_uploader_progress_queue")
                logger.info("Connected to RabbitMQ.")
                break
            except AMQPConnectionError:
                logger.warning("RabbitMQ not available, retrying in 5 seconds...")
                time.sleep(5)

    def process_and_save_image(self, body: bytes, image_type: str) -> None:
        try:
            message = json.loads(body)
            image_data = base64.b64decode(message["image_data"])

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

            if image_type == "raw":
                self.uploaded_raw_images += 1
                self.update_progress("raw")
            elif image_type == "processed":
                self.uploaded_processed_images += 1
                self.update_progress("processed")

        except Exception as e:
            logger.error(f"Error saving image: {e}")

    def process_and_save_metrics(self, body: bytes) -> None:
        try:
            message = json.loads(body)
            self.metrics_collection.update_one(
                {"training_id": message["training_id"]},
                {"$push": {"epochs": message}},
                upsert=True,
            )
            logger.info(f"Saved metrics for training ID: {message['training_id']}")
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")

    def raw_callback(self, _ch, _method, _properties, body) -> None:
        self.process_and_save_image(body, "raw")

    def processed_callback(self, _ch, _method, _properties, body) -> None:
        self.process_and_save_image(body, "processed")

    def metrics_callback(self, _ch, _method, _properties, body) -> None:
        logger.info("Received metrics")
        self.process_and_save_metrics(body)

    def update_progress(self, image_type: str) -> None:
        """
        Sends progress update to the uploader progress queue based on image type.
        """
        if image_type == "raw":
            progress_message = {
                "uploaded": self.uploaded_raw_images,
                "total": self.total_raw_images,
            }
            self.channel.basic_publish(
                exchange="",
                routing_key="raw_uploader_progress_queue",
                body=json.dumps(progress_message),
            )
            logger.info(f"Sent raw upload progress update: {progress_message}")
        elif image_type == "processed":
            progress_message = {
                "uploaded": self.uploaded_processed_images,
                "total": self.total_processed_images,
            }
            self.channel.basic_publish(
                exchange="",
                routing_key="processed_uploader_progress_queue",
                body=json.dumps(progress_message),
            )
            logger.info(f"Sent processed upload progress update: {progress_message}")

    def start_consuming(self) -> None:
        while True:
            try:
                self.total_raw_images = self.get_queue_length(
                    "raw_image_queue_uploader"
                )  # Set total raw images at start
                self.total_processed_images = self.get_queue_length(
                    "processed_image_queue"
                )  # Set total processed images at start

                self.channel.basic_consume(
                    queue="raw_image_queue_uploader",
                    on_message_callback=self.raw_callback,
                    auto_ack=True,
                )
                self.channel.basic_consume(
                    queue="processed_image_queue",
                    on_message_callback=self.processed_callback,
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

    def get_queue_length(self, queue_name) -> int:
        """
        Retrieves the length of the queue.

        Parameters
        ----------
        queue_name : str
            Name of the queue.

        Returns
        -------
        int
            Number of messages in the queue.
        """
        queue_state = self.channel.queue_declare(queue=queue_name, passive=True)
        return queue_state.method.message_count


if __name__ == "__main__":
    uploader = ImageUploader()
    uploader.start_consuming()
