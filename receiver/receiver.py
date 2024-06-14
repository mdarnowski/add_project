"""
ImageDownloader Module
=======================

This module provides the `ImageDownloader` class for downloading images from a MongoDB database using GridFS
and sending them to a RabbitMQ queue in response to requests from the Trainer.

Classes
-------
- ImageDownloader

Dependencies
------------
- base64
- json
- os
- time
- gridfs
- pika
- pymongo
- loguru

Environment Variables
---------------------
- MONGO_HOST : str, optional
    The MongoDB URI (default: "mongodb://localhost:27017/")
- RABBITMQ_HOST : str, optional
    The RabbitMQ host address (default: "localhost")
"""

import base64
import json
import os
import time

import gridfs
import pika
from loguru import logger
from pika.exceptions import AMQPConnectionError, ConnectionClosed
from pymongo import MongoClient


class ImageDownloader:
    """
    ImageDownloader Class
    =====================

    This class provides functionality to download images from a MongoDB database
    using GridFS and to handle RabbitMQ message queues for receiving requests
    and sending image data.

    Methods
    -------
    __init__()
        Initializes the ImageDownloader instance, sets up MongoDB and RabbitMQ connections.
    _fetch_images(split: str)
        Fetches images from the MongoDB database for the specified dataset split.
    _send_message(message: dict)
        Sends a message to the RabbitMQ queue.
    handle_request(ch, method, properties, body)
        Callback function for handling data requests from the trainer.
    start_consuming()
        Starts consuming request messages from RabbitMQ.
    send_num_classes()
        Sends the number of classes to a separate queue.
    """

    def __init__(self) -> None:
        """
        Initializes the ImageDownloader instance.

        Sets up the MongoDB and RabbitMQ connections and declares queues.

        Environment Variables
        ---------------------
        - MONGO_HOST : str, optional
            The MongoDB URI (default: "mongodb://localhost:27017/")
        - RABBITMQ_HOST : str, optional
            The RabbitMQ host address (default: "localhost")
        """
        self.mongo_uri = os.getenv("MONGO_HOST", "mongodb://localhost:27017/")
        self.rabbitmq_host = os.getenv("RABBITMQ_HOST", "localhost")
        self.client = MongoClient(self.mongo_uri)
        self.db = self.client.bird_dataset
        self.fs = gridfs.GridFS(self.db)
        self.images_collection = self.db.images
        self.connection = None
        self.channel = None
        self.connect_to_rabbitmq()

    def connect_to_rabbitmq(self) -> None:
        """
        Establishes a connection to RabbitMQ and declares the required queues.

        Retries the connection every 5 seconds if RabbitMQ is not available.
        """
        while True:
            try:
                self.connection = pika.BlockingConnection(
                    pika.ConnectionParameters(self.rabbitmq_host)
                )
                self.channel = self.connection.channel()
                self.channel.queue_declare(queue="image_queue_downloader")
                self.channel.queue_declare(queue="data_request_queue")
                self.channel.queue_declare(
                    queue="num_classes_queue"
                )  # Added for num_classes
                logger.info("Connected to RabbitMQ.")
                break
            except AMQPConnectionError:
                logger.warning("RabbitMQ not available, retrying in 5 seconds...")
                time.sleep(5)

    def _fetch_images(self, split: str) -> list:
        """
        Fetches images from the MongoDB database for the specified dataset split.

        Parameters
        ----------
        split : str
            The dataset split (e.g., 'train', 'val', 'test').

        Returns
        -------
        list
            A list of dictionaries containing image metadata and encoded data.
        """
        image_documents = self.images_collection.find(
            {"set_type": split, "image_type": "processed"}
        )
        images = []
        for doc in image_documents:
            try:
                image_data = self.fs.get(doc["image_id"]).read()
                encoded_data = base64.b64encode(image_data).decode("utf-8")
                image_dict = {
                    "image_data": encoded_data,
                    "image_path": doc.get(
                        "filename", "unknown"
                    ),  # Use "unknown" if filename is missing
                    "species": doc["species"],
                    "split": doc["set_type"],
                    "label": doc["label"],
                }
                images.append(image_dict)
            except KeyError as e:
                logger.error(f"Document missing field {e}: {doc}")
            except Exception as e:
                logger.error(f"Error processing document: {e}")
        return images

    def _send_message(self, message: dict) -> None:
        """
        Sends a message to the RabbitMQ queue.

        Parameters
        ----------
        message : dict
            The message to send.
        """
        try:
            self.channel.basic_publish(
                exchange="",
                routing_key="image_queue_downloader",
                body=json.dumps(message),
            )
            logger.info(
                f"Sent message for image {message.get('image_path', 'unknown')}"
            )
        except Exception as e:
            logger.error(f"Error sending message: {e}")

    def handle_request(self, ch, method, properties, body) -> None:
        """
        Callback function for handling data requests from the trainer.

        Parameters
        ----------
        ch : channel
            The channel object.
        method : method
            The method object.
        properties : properties
            The properties object.
        body : bytes
            The message body.
        """
        try:
            request = json.loads(body)
            split = request["split"]
            logger.info(f"Received request for {split} split")
            images = self._fetch_images(split)
            for image in images:
                self._send_message(image)
            # Send a "done" message to indicate that all data has been sent
            self._send_message({"status": "done", "split": split})
            # Send num_classes after sending all images
            self.send_num_classes()
        except Exception as e:
            logger.error(f"Error processing request: {e}")

    def send_num_classes(self) -> None:
        """
        Sends the number of classes to the num_classes_queue.
        """
        try:
            num_classes = len(
                self.images_collection.distinct("label", {"image_type": "processed"})
            )
            message = {"num_classes": num_classes}
            self.channel.basic_publish(
                exchange="", routing_key="num_classes_queue", body=json.dumps(message)
            )
            logger.info(f"Sent num_classes: {num_classes}")
        except Exception as e:
            logger.error(f"Error sending num_classes: {e}")

    def start_consuming(self) -> None:
        """
        Starts consuming request messages from RabbitMQ.

        Consumes messages from the 'data_request_queue' queue and handles them.
        """
        while True:
            try:
                self.channel.basic_consume(
                    queue="data_request_queue",
                    on_message_callback=self.handle_request,
                    auto_ack=True,
                )
                logger.info("ImageDownloader is listening for requests...")
                self.channel.start_consuming()
            except (ConnectionClosed, AMQPConnectionError):
                logger.warning("Connection lost, reconnecting...")
                self.connect_to_rabbitmq()


if __name__ == "__main__":
    downloader = ImageDownloader()
    downloader.start_consuming()
