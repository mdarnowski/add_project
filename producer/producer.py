import base64
import json
import os
import random
import time

import pika
from loguru import logger

# Configuration
DATASET_PATH = "/app/cub_200_2011/images"
RAW_IMAGE_QUEUE = "raw_image_queue"
RAW_IMAGE_QUEUE_FOR_UPLOADER = "raw_image_queue_uploader"


def connect_to_rabbitmq() -> pika.BlockingConnection:
    while True:
        try:
            connection = pika.BlockingConnection(pika.ConnectionParameters("rabbitmq"))
            logger.info("Connected to RabbitMQ.")
            return connection
        except pika.exceptions.AMQPConnectionError:
            logger.warning("RabbitMQ not available, retrying in 5 seconds...")
            time.sleep(5)


def publish_images_to_queue(
    dataset_path: str, queue_name: str, queue_name_uploader: str
) -> None:
    connection = connect_to_rabbitmq()
    channel = connection.channel()
    channel.queue_declare(queue=queue_name)
    channel.queue_declare(queue=queue_name_uploader)

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

                    parts = os.path.basename(subdir).split(".")

                    images.append(
                        {
                            "image_path": os.path.basename(image_path),
                            "image_data": encoded_image,
                            "label": int(parts[0]) - 1,
                            "species": parts[1],
                        }
                    )
                except Exception as e:
                    logger.error(f"Failed to process {image_path}. Error: {e}")

        if images:
            # Shuffle and split the dataset for the current subdirectory
            random.shuffle(images)
            total_data = len(images)
            train_split = int(total_data * 0.7)
            val_split = int(total_data * 0.1)

            train_data = images[:train_split]
            val_data = images[train_split : train_split + val_split]
            test_data = images[train_split + val_split :]

            def publish_to_queues(data, split):
                data["split"] = split
                channel.basic_publish(
                    exchange="", routing_key=queue_name, body=json.dumps(data)
                )
                channel.basic_publish(
                    exchange="", routing_key=queue_name_uploader, body=json.dumps(data)
                )

            for data in train_data:
                publish_to_queues(data, "train")
            for data in val_data:
                publish_to_queues(data, "val")
            for data in test_data:
                publish_to_queues(data, "test")

    connection.close()
    logger.info("Finished publishing images.")


if __name__ == "__main__":
    publish_images_to_queue(DATASET_PATH, RAW_IMAGE_QUEUE, RAW_IMAGE_QUEUE_FOR_UPLOADER)
