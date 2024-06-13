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
    def __init__(self):
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
        if self.connection is not None and self.connection.is_open:
            return
        logger.info("Reconnecting to RabbitMQ...")
        self.connect_to_rabbitmq()

    def declare_queues(self) -> None:
        for queue_name in [
            self.queue_name,
            self.queue_name_uploader,
            self.progress_queue,
            self.trigger_queue,
        ]:
            self.channel.queue_declare(queue=queue_name)

    def consume_trigger(self) -> None:
        self.channel.basic_consume(
            queue=self.trigger_queue,
            on_message_callback=self.process_images,
            auto_ack=True,
        )
        logger.info("Waiting for trigger to start processing images...")
        self.channel.start_consuming()

    def process_images(self, _ch, _method, _properties, _body) -> None:
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
        return sum(
            len(files)
            for _, _, files in os.walk(self.dataset_path)
            if any(file.endswith(".jpg") for file in files)
        )

    def publish_images(self, subdir, files) -> None:
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

    def publish_to_queues(self, data, split) -> None:
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
                    {"processed": self.processed_images, "total": self.total_images}
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
