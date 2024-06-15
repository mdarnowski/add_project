"""
Producer Module
==============
Module for image processing and message queueing using RabbitMQ.

This module contains the `ImageProducer` class, which is responsible for:
- Connecting to RabbitMQ
- Consuming messages from a trigger queue
- Processing images from a dataset
- Publishing image data to various queues

It is designed to handle image data in a specified directory, encode it, and
distribute it across different queues for further processing.

Dependencies:
    - base64
    - json
    - os
    - random
    - time
    - threading
    - pika
    - loguru

Environment Variables:
    - DATASET_PATH: Path to the directory containing images.
    - RAW_IMAGE_QUEUE: Name of the queue to publish raw image data.
    - RAW_IMAGE_QUEUE_FOR_UPLOADER: Name of the uploader queue.
    - PROGRESS_QUEUE: Name of the queue to publish progress information.
    - TRIGGER_QUEUE: Name of the queue to listen for trigger messages.
    - RABBITMQ_HOST: Hostname of the RabbitMQ server.
"""

import base64
import json
import os
import random
import time
from threading import Lock

import pika
from loguru import logger
from pika.exceptions import AMQPConnectionError, ConnectionClosed


class ImageProducer:
    """
    A class used to produce and publish image data to RabbitMQ queues.

    The `ImageProducer` class is designed to:
    - Connect to RabbitMQ
    - Consume trigger messages from a queue
    - Process image files from a specified dataset path
    - Publish the processed image data to designated queues

    Attributes
    ----------
    dataset_path : str
        Path to the image dataset directory.
    queue_name : str
        Name of the RabbitMQ queue for raw images.
    queue_name_uploader : str
        Name of the RabbitMQ queue for image uploader.
    progress_queue : str
        Name of the RabbitMQ queue for progress updates.
    trigger_queue : str
        Name of the RabbitMQ queue for trigger messages.
    rabbitmq_host : str
        Hostname of the RabbitMQ server.
    lock : threading.Lock
        A lock object to control access to image processing.
    processing : bool
        Flag to indicate if image processing is currently ongoing.
    total_images : int
        Total number of images to process.
    processed_images : int
        Number of images processed so far.
    last_progress_report : float
        Last reported progress percentage.
    connection : pika.BlockingConnection
        Connection object for RabbitMQ.
    channel : pika.adapters.blocking_connection.BlockingChannel
        Channel object for RabbitMQ.

    Methods
    -------
    connect_to_rabbitmq() -> None
        Establishes a connection to the RabbitMQ server and declares the queues.
    reconnect_to_rabbitmq() -> None
        Re-establishes the RabbitMQ connection if it is closed.
    declare_queues() -> None
        Declares the necessary RabbitMQ queues.
    consume_trigger() -> None
        Consumes messages from the trigger queue and starts image processing.
    process_images(_ch, _method, _properties, _body) -> None
        Processes images and publishes them to queues.
    count_images() -> int
        Counts the total number of images to be processed.
    publish_images(subdir: str, files: list) -> None
        Encodes and publishes images from a subdirectory.
    publish_to_queues(data: dict, split: str) -> None
        Publishes image data to the appropriate queues.
    start() -> None
        Starts the image producer to listen for trigger messages.
    """

    def __init__(self):
        """
        Initializes the ImageProducer with default values and connects to RabbitMQ.

        Environment variables are used to set the paths and queue names.
        If not set, defaults are used.
        """
        self.dataset_path = os.getenv("DATASET_PATH", "/app/cub_200_2011/images")
        self.queue_name = os.getenv("RAW_IMAGE_QUEUE", "raw_image_queue")
        self.queue_name_uploader = os.getenv(
            "RAW_IMAGE_QUEUE_FOR_UPLOADER", "raw_image_queue_uploader"
        )
        self.progress_queue = os.getenv("PROGRESS_QUEUE", "progress_queue")
        self.trigger_queue = os.getenv("TRIGGER_QUEUE", "trigger_queue")
        self.rabbitmq_host = os.getenv("RABBITMQ_HOST", "rabbitmq")

        self.lock = Lock()
        self.processing = False
        self.total_images = 0
        self.processed_images = 0
        self.last_progress_report = 0
        self.connection = None
        self.channel = None
        self.connect_to_rabbitmq()

    def connect_to_rabbitmq(self) -> None:
        """
        Establishes a connection to RabbitMQ and declares the required queues.

        This method attempts to connect to RabbitMQ indefinitely until a
        connection is established.
        """
        while True:
            try:
                self.connection = pika.BlockingConnection(
                    pika.ConnectionParameters(self.rabbitmq_host)
                )
                self.channel = self.connection.channel()
                self.declare_queues()
                logger.info("Connected to RabbitMQ.")
                break
            except AMQPConnectionError:
                logger.warning("RabbitMQ not available, retrying in 5 seconds...")
                time.sleep(5)

    def reconnect_to_rabbitmq(self) -> None:
        """
        Reconnects to RabbitMQ if the current connection is closed.

        This method ensures the connection to RabbitMQ is always open
        before performing operations.
        """
        if self.connection is not None and self.connection.is_open:
            return
        logger.info("Reconnecting to RabbitMQ...")
        self.connect_to_rabbitmq()

    def declare_queues(self) -> None:
        """
        Declares the required RabbitMQ queues.

        This method declares the queues for raw images, uploader, progress,
        and trigger based on environment variable names.
        """
        for queue_name in [
            self.queue_name,
            self.queue_name_uploader,
            self.progress_queue,
            self.trigger_queue,
        ]:
            self.channel.queue_declare(queue=queue_name)

    def consume_trigger(self) -> None:
        """
        Consumes trigger messages from the trigger queue.

        This method starts consuming messages from the trigger queue and
        processes images upon receiving a message.
        """
        self.channel.basic_consume(
            queue=self.trigger_queue,
            on_message_callback=self.process_images,
            auto_ack=True,
        )
        logger.info("Waiting for trigger to start processing images...")
        self.channel.start_consuming()

    def process_images(self, _ch, _method, _properties, _body) -> None:
        """
        Processes images from the dataset path and publishes them to queues.

        Parameters
        ----------
        _ch : pika.adapters.blocking_connection.BlockingChannel
            The channel object (unused).
        _method : pika.spec.Basic.Deliver
            The method frame (unused).
        _properties : pika.spec.BasicProperties
            The properties (unused).
        _body : bytes
            The message body (unused).
        """
        with self.lock:
            if self.processing:
                logger.warning("Already processing images. Skipping.")
                return
            self.processing = True
        try:
            self.total_images = self.count_images()
            self.processed_images = 0
            self.last_progress_report = 0

            for subdir, _, files in os.walk(self.dataset_path):
                images_to_process = [file for file in files if file.endswith(".jpg")]
                if images_to_process:
                    self.publish_images(subdir, images_to_process)
        finally:
            with self.lock:
                self.processing = False

    def count_images(self) -> int:
        """
        Counts the total number of images in the dataset path.

        Returns
        -------
        int
            The total number of image files found.
        """
        return sum(
            len(files)
            for _, _, files in os.walk(self.dataset_path)
            if any(file.endswith(".jpg") for file in files)
        )

    def publish_images(self, subdir: str, files: list) -> None:
        """
        Encodes and publishes images from a subdirectory.

        Parameters
        ----------
        subdir : str
            The directory path containing image files.
        files : list
            List of image file names to be processed.
        """

        def encode_image(file):
            image_path = os.path.join(subdir, file)
            try:
                with open(image_path, "rb") as image_file:
                    encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
                parts = os.path.basename(subdir).split(".")
                return {
                    "image_path": os.path.basename(image_path),
                    "image_data": encoded_image,
                    "label": int(parts[0]) - 1,
                    "species": parts[1],
                }
            except Exception as e:
                logger.error(f"Failed to process {image_path}. Error: {e}")
                return None

        images = [encode_image(file) for file in files]

        if images:
            random.shuffle(images)
            total_data = len(images)
            train_split = int(total_data * 0.7)
            val_split = int(total_data * 0.1)

            splits = [
                ("train", images[:train_split]),
                ("val", images[train_split : train_split + val_split]),
                ("test", images[train_split + val_split :]),
            ]

            for split, split_data in splits:
                for data in split_data:
                    self.publish_to_queues(data, split)

    def publish_to_queues(self, data: dict, split: str) -> None:
        """
        Publishes image data to the appropriate queues.

        Parameters
        ----------
        data : dict
            The image data to be published.
        split : str
            The split category (train/val/test) for the data.
        """
        self.processed_images += 1
        progress_percentage = (self.processed_images / self.total_images) * 100
        if (
            progress_percentage - self.last_progress_report >= 1
            or self.processed_images == self.total_images
        ):
            logger.info(
                f"Processed {self.processed_images}/{self.total_images} images."
            )
            self.channel.basic_publish(
                exchange="",
                routing_key=self.progress_queue,
                body=json.dumps(
                    {"produced": self.processed_images, "total": self.total_images}
                ),
            )
            self.last_progress_report = progress_percentage

        data["split"] = split
        self.channel.basic_publish(
            exchange="", routing_key=self.queue_name, body=json.dumps(data)
        )
        self.channel.basic_publish(
            exchange="", routing_key=self.queue_name_uploader, body=json.dumps(data)
        )

    def start(self) -> None:
        """
        Starts the image producer to listen for trigger messages.

        This method runs indefinitely, reconnecting to RabbitMQ if the
        connection is lost.
        """
        while True:
            try:
                self.reconnect_to_rabbitmq()
                self.consume_trigger()
            except (ConnectionClosed, AMQPConnectionError):
                logger.warning("Connection lost, reconnecting...")
                time.sleep(5)


if __name__ == "__main__":
    producer = ImageProducer()
    producer.start()
