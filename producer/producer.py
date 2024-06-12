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
PROGRESS_QUEUE = "progress_queue"


def connect_to_rabbitmq() -> pika.BlockingConnection:
    while True:
        try:
            connection = pika.BlockingConnection(pika.ConnectionParameters("rabbitmq"))
            logger.info("Connected to RabbitMQ.")
            return connection
        except pika.exceptions.AMQPConnectionError:
            logger.warning("RabbitMQ not available, retrying in 5 seconds...")
            time.sleep(5)


def publish_to_queues(
    data,
    split,
    channel,
    queue_name,
    queue_name_uploader,
    progress_queue,
    processed_images,
    total_images,
):
    data["split"] = split
    channel.basic_publish(exchange="", routing_key=queue_name, body=json.dumps(data))
    channel.basic_publish(
        exchange="", routing_key=queue_name_uploader, body=json.dumps(data)
    )
    processed_images += 1
    logger.info(f"Processed {processed_images}/{total_images} images.")
    channel.basic_publish(
        exchange="",
        routing_key=progress_queue,
        body=json.dumps({"processed": processed_images, "total": total_images}),
    )
    return processed_images


def process_and_publish_images(
    subdir,
    files,
    channel,
    queue_name,
    queue_name_uploader,
    progress_queue,
    processed_images,
    total_images,
):
    def process_image(file):
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

    images = [process_image(file) for file in files if file.endswith(".jpg")]
    images = [image for image in images if image]  # Filter out None values

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
                processed_images = publish_to_queues(
                    data,
                    split,
                    channel,
                    queue_name,
                    queue_name_uploader,
                    progress_queue,
                    processed_images,
                    total_images,
                )

    return processed_images


def publish_images_to_queue(
    dataset_path: str, queue_name: str, queue_name_uploader: str, progress_queue: str
) -> None:
    connection = connect_to_rabbitmq()
    channel = connection.channel()
    channel.queue_declare(queue=queue_name)
    channel.queue_declare(queue=queue_name_uploader)
    channel.queue_declare(queue=progress_queue)

    total_images = sum(
        [
            len(files)
            for _, _, files in os.walk(dataset_path)
            if any(file.endswith(".jpg") for file in files)
        ]
    )

    processed_images = 0
    for subdir, _, files in os.walk(dataset_path):
        processed_images = process_and_publish_images(
            subdir,
            files,
            channel,
            queue_name,
            queue_name_uploader,
            progress_queue,
            processed_images,
            total_images,
        )

    connection.close()
    logger.info("Finished publishing images.")


if __name__ == "__main__":
    publish_images_to_queue(
        DATASET_PATH, RAW_IMAGE_QUEUE, RAW_IMAGE_QUEUE_FOR_UPLOADER, PROGRESS_QUEUE
    )
