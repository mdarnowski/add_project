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

# Environment variables
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
RAW_IMAGE_QUEUE = os.getenv("RAW_IMAGE_QUEUE", "raw_image_queue")
PROCESSED_IMAGE_QUEUE = os.getenv("PROCESSED_IMAGE_QUEUE", "processed_image_queue")
PROCESSOR_PROGRESS_QUEUE = os.getenv(
    "PROCESSOR_PROGRESS_QUEUE", "processor_progress_queue"
)


class RabbitMQProcessor:
    """
    RabbitMQProcessor handles image processing and communication with RabbitMQ queues.

    Attributes
    ----------
    rabbitmq_host : str
        Hostname for RabbitMQ server.
    raw_image_queue : str
        Queue name for raw images.
    processed_image_queue : str
        Queue name for processed images.
    processor_progress_queue : str
        Queue name for processor progress updates.
    connection : pika.BlockingConnection
        Connection object for RabbitMQ.
    channel : pika.channel.Channel
        Channel object for RabbitMQ.
    processed_images : int
        Count of processed images.

    Methods
    -------
    connect_to_rabbitmq():
        Establishes connection to RabbitMQ server.
    transform_image(image_data, split):
        Transforms the input image data according to the split type.
    send_to_queue(data: dict):
        Processes the image and sends the transformed image to the processed queue.
    update_progress():
        Sends progress update to the processor progress queue.
    reset_counter():
        Resets the processed images counter.
    start_consuming():
        Starts consuming messages from the raw image queue.
    """

    def __init__(self):
        """
        Initializes the RabbitMQProcessor class.

        Sets up the RabbitMQ host, raw image queue, processed image queue,
        and processor progress queue from environment variables or defaults.
        """
        self.rabbitmq_host = RABBITMQ_HOST
        self.raw_image_queue = RAW_IMAGE_QUEUE
        self.processed_image_queue = PROCESSED_IMAGE_QUEUE
        self.processor_progress_queue = PROCESSOR_PROGRESS_QUEUE
        self.connection = None
        self.channel = None
        self.processed_images = 0
        self.connect_to_rabbitmq()

    def connect_to_rabbitmq(self):
        """
        Connects to RabbitMQ server and declares the processed image and processor progress queues.

        Retries connection every 5 seconds if initial connection fails.
        """
        while self.connection is None or self.connection.is_closed:
            try:
                self.connection = pika.BlockingConnection(
                    pika.ConnectionParameters(self.rabbitmq_host)
                )
                self.channel = self.connection.channel()
                self.channel.queue_declare(queue=self.processed_image_queue)
                self.channel.queue_declare(queue=self.processor_progress_queue)
                logger.info("Connected to RabbitMQ.")
            except (AMQPConnectionError, ConnectionClosed):
                logger.warning("RabbitMQ not available or connection closed, retrying in 5 seconds...")
                time.sleep(5)
                self.connection = None
                self.channel = None

    def transform_image(self, image_data, split):
        """
        Transforms the input image data based on the split type.

        Parameters
        ----------
        image_data : bytes
            Base64 decoded image data.
        split : str
            The dataset split type. Can be 'train', 'val', or 'test'.

        Returns
        -------
        numpy.ndarray
            The transformed image array.
        """
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        img_array = np.array(image)

        if split == "train":
            img_array = tf.image.resize(img_array, [299, 299])
        else:
            img_array = tf.image.resize(img_array, [299, 299])

        img_array = tf.cast(img_array, tf.float32) / 255.0
        return img_array

    def send_to_queue(self, data: dict):
        """
        Processes the image and sends the transformed image to the processed queue.

        Parameters
        ----------
        data : dict
            Dictionary containing image data, path, label, split, and species.
        """
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

            # Update progress
            self.processed_images += 1
            self.update_progress()

        except Exception as e:
            logger.error(f"Error processing image. Error: {e}")
            if isinstance(e, AMQPConnectionError) or isinstance(e, ConnectionClosed):
                logger.error("Connection lost, attempting to reconnect...")
                self.connect_to_rabbitmq()

    def update_progress(self):
        """
        Sends progress update to the processor progress queue.
        """
        progress_message = {"processed": self.processed_images}
        self.channel.basic_publish(
            exchange="",
            routing_key=self.processor_progress_queue,
            body=json.dumps(progress_message),
        )
        logger.info(f"Sent progress update: {progress_message}")

    def reset_counter(self):
        """
        Resets the processed images counter.
        """
        self.processed_images = 0
        logger.info("Processed images counter reset to 0")

    def start_consuming(self):
        """
        Starts consuming messages from the raw image queue and processes them.

        Declares the raw image queue and sets up a callback to handle messages.
        Retries consuming if the connection is lost.
        """
        while True:
            try:
                if self.channel is None or self.channel.is_closed:
                    self.connect_to_rabbitmq()

                self.channel.queue_declare(queue=self.raw_image_queue)

                def callback(ch, method, properties, body):
                    message = json.loads(body)
                    if message.get("reset"):
                        self.reset_counter()
                    else:
                        self.send_to_queue(message)
                        logger.info(
                            f"Processed and sent message for image_path: {message['image_path']}"
                        )

                self.channel.basic_consume(
                    queue=self.raw_image_queue, on_message_callback=callback, auto_ack=True
                )
                logger.info("Processor is listening for messages...")
                self.channel.start_consuming()
            except (AMQPConnectionError, ConnectionClosed) as e:
                logger.error(f"Connection lost or closed. Reconnecting... Error: {e}")
                self.connection = None
                self.channel = None
                time.sleep(5)


if __name__ == "__main__":
    processor = RabbitMQProcessor()
    processor.start_consuming()
