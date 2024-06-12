import base64
import json
import os
import random
import time

import pika
from loguru import logger

# Configuration
DATASET_PATH = "/app/cub_200_2011/images"
RABBITMQ_QUEUE = "raw_image_queue"


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


def publish_images_to_queue(dataset_path: str, queue_name: str) -> None:
    """
    Publish image data from a dataset directory to the RabbitMQ queue.
    """
    connection = connect_to_rabbitmq()
    channel = connection.channel()
    channel.queue_declare(queue=queue_name)

    for subdir, _, files in os.walk(dataset_path):

        images = []
        for file in files:
            if file.endswith(".jpg"):
                image_path = os.path.join(subdir, file)
                try:
                    with open(image_path, "rb") as image_file:
                        encoded_image = base64.b64encode(image_file.read()).decode(
                            "utf-8"
                        )
                    images.append(
                        {"image_path": image_path, "image_data": encoded_image}
                    )
                except Exception as e:
                    logger.error(f"Failed to process {image_path}. Error: {e}")

        # Shuffle and split the dataset
        random.shuffle(images)
        total_data = len(images)
        train_split = int(total_data * 0.7)
        val_split = int(total_data * 0.1)

        train_data = images[:train_split]
        val_data = images[train_split : train_split + val_split]
        test_data = images[train_split + val_split :]

        # Send data to respective queues
        for data in train_data:
            data["split"] = "train"
            channel.basic_publish(
                exchange="", routing_key=queue_name, body=json.dumps(data)
            )
        for data in val_data:
            data["split"] = "val"
            channel.basic_publish(
                exchange="", routing_key=queue_name, body=json.dumps(data)
            )
        for data in test_data:
            data["split"] = "test"
            channel.basic_publish(
                exchange="", routing_key=queue_name, body=json.dumps(data)
            )

    connection.close()
    logger.info("Finished publishing images.")


if __name__ == "__main__":
    publish_images_to_queue(DATASET_PATH, RABBITMQ_QUEUE)
