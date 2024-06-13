import base64
import io
import json
import os
import time

import pika
import numpy as np
from loguru import logger
from pika.exceptions import AMQPConnectionError, ConnectionClosed
from PIL import Image
import tensorflow as tf


class RabbitMQProcessor:
    def __init__(self):
        self.rabbitmq_host = os.getenv("RABBITMQ_HOST", "rabbitmq")
        self.raw_image_queue = os.getenv("RAW_IMAGE_QUEUE", "raw_image_queue")
        self.processed_image_queue = os.getenv(
            "PROCESSED_IMAGE_QUEUE", "processed_image_queue"
        )
        self.connection = None
        self.channel = None
        self.connect_to_rabbitmq()

    def connect_to_rabbitmq(self):
        while self.connection is None or self.connection.is_closed:
            try:
                self.connection = pika.BlockingConnection(
                    pika.ConnectionParameters(self.rabbitmq_host)
                )
                self.channel = self.connection.channel()
                self.channel.queue_declare(queue=self.processed_image_queue)
                logger.info("Connected to RabbitMQ.")
            except AMQPConnectionError:
                logger.warning("RabbitMQ not available, retrying in 5 seconds...")
                time.sleep(5)

    def transform_image(self, image_data, split):
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        img_array = np.array(image)

        if split == "train":
            img_array = tf.image.resize(img_array, [299, 299])
        else:  # val or test
            img_array = tf.image.resize(img_array, [299, 299])

        img_array = tf.cast(img_array, tf.float32) / 255.0

        return img_array

    def send_to_queue(self, data: dict):
        try:
            if self.channel is None or self.channel.is_closed:
                self.connect_to_rabbitmq()

            image_path = data["image_path"]
            image_data = base64.b64decode(data["image_data"])

            # Transform image
            img_array = self.transform_image(image_data, data["split"])

            # Convert back to bytes
            img_array = (img_array * 255).numpy().astype(np.uint8)
            image = Image.fromarray(img_array)
            image_bytes = io.BytesIO()
            image.save(image_bytes, format="JPEG")

            processed_message = {
                "image_path": image_path,
                "image_data": base64.b64encode(image_bytes.getvalue()).decode("utf-8"),
                "label": data["label"],
                "split": data["split"],
                "species": data["species"],
            }

            self.channel.basic_publish(
                exchange="",
                routing_key=self.processed_image_queue,
                body=json.dumps(processed_message),
            )
            logger.info(f"Sent processed message for image_path: {image_path} to queue")

        except Exception as e:
            logger.error(f"Error processing image. Error: {e}")
            if isinstance(e, AMQPConnectionError) or isinstance(e, ConnectionClosed):
                logger.error("Connection lost, attempting to reconnect...")
                self.connect_to_rabbitmq()

    def start_consuming(self):
        if self.channel is None or self.channel.is_closed:
            self.connect_to_rabbitmq()

        self.channel.queue_declare(queue=self.raw_image_queue)

        def callback(ch, method, properties, body):
            message = json.loads(body)
            logger.info(f"Received message for image_path: {message['image_path']}")
            self.send_to_queue(message)
            logger.info(
                f"Processed and sent message for image_path: {message['image_path']}"
            )

        self.channel.basic_consume(
            queue=self.raw_image_queue, on_message_callback=callback, auto_ack=True
        )
        logger.info("Processor is listening for messages...")
        self.channel.start_consuming()


if __name__ == "__main__":
    processor = RabbitMQProcessor()
    processor.start_consuming()
