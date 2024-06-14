"""
ImageUploader Module
====================

This module provides the `ImageUploader` class for uploading images to a MongoDB database using GridFS
and handling message queues with RabbitMQ.

Classes
-------
- ImageUploader

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
import pymongo
from loguru import logger
from pika.exceptions import AMQPConnectionError, ConnectionClosed
from pymongo import MongoClient


class ImageUploader:
    """
    ImageUploader Class
    ===================

    This class provides functionality to upload images to a MongoDB database
    using GridFS and to handle RabbitMQ message queues for processing image data.

    Methods
    -------
    __init__()
        Initializes the ImageUploader instance, sets up MongoDB and RabbitMQ connections.
    _create_indexes()
        Creates necessary indexes on the images collection in MongoDB.
    connect_to_rabbitmq()
        Establishes a connection to RabbitMQ and declares required queues.
    process_and_save(body: bytes, image_type: str)
        Decodes the image data, saves it to GridFS, and inserts metadata into the images collection.
    raw_callback(_ch, _method, _properties, body)
        Callback function to process raw image data messages.
    processed_callback(_ch, _method, _properties, body)
        Callback function to process processed image data messages.
    start_consuming()
        Starts consuming messages from the RabbitMQ queues.
    """

    def __init__(self) -> None:
        """
        Initializes the ImageUploader instance.

        Sets up the MongoDB and RabbitMQ connections, creates indexes, and declares queues.

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
        self._create_indexes()
        self.connection = None
        self.channel = None
        self.connect_to_rabbitmq()

    def _create_indexes(self) -> None:
        """
        Creates indexes on the images collection in MongoDB.

        Indexes
        -------
        - species : ascending
        - set_type : ascending
        - image_type : ascending
        - label : ascending
        """
        self.images_collection.create_index([("species", pymongo.ASCENDING)])
        self.images_collection.create_index([("set_type", pymongo.ASCENDING)])
        self.images_collection.create_index([("image_type", pymongo.ASCENDING)])
        self.images_collection.create_index([("label", pymongo.ASCENDING)])

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
                self.channel.queue_declare(queue="raw_image_queue_uploader")
                self.channel.queue_declare(queue="processed_image_queue")
                logger.info("Connected to RabbitMQ.")
                break
            except AMQPConnectionError:
                logger.warning("RabbitMQ not available, retrying in 5 seconds...")
                time.sleep(5)

    def process_and_save(self, body: bytes, image_type: str) -> None:
        """
        Processes and saves the image data to GridFS and inserts metadata into the images collection.

        Parameters
        ----------
        body : bytes
            The message body containing the image data.
        image_type : str
            The type of the image (e.g., 'raw' or 'processed').

        Logs
        ----
        - Info : on successful save
        - Error : if there is an error during save
        """
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
        except Exception as e:
            logger.error(f"Error saving image: {e}")

    def raw_callback(self, _ch, _method, _properties, body) -> None:
        """
        Callback function for processing raw image data messages.

        Parameters
        ----------
        _ch : channel
            The channel object.
        _method : method
            The method object.
        _properties : properties
            The properties object.
        body : bytes
            The message body.
        """
        self.process_and_save(body, "raw")

    def processed_callback(self, _ch, _method, _properties, body) -> None:
        """
        Callback function for processing processed image data messages.

        Parameters
        ----------
        _ch : channel
            The channel object.
        _method : method
            The method object.
        _properties : properties
            The properties object.
        body : bytes
            The message body.
        """
        self.process_and_save(body, "processed")

    def start_consuming(self) -> None:
        """
        Starts consuming messages from the RabbitMQ queues.

        Consumes messages from both 'raw_image_queue_uploader' and 'processed_image_queue' and
        processes them using the respective callback functions.

        Retries the connection if it is lost.
        """
        while True:
            try:
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
                logger.info("Uploader is listening for messages...")
                self.channel.start_consuming()
            except (ConnectionClosed, AMQPConnectionError):
                logger.warning("Connection lost, reconnecting...")
                self.connect_to_rabbitmq()


if __name__ == "__main__":
    uploader = ImageUploader()
    uploader.start_consuming()
