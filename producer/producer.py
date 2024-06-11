import base64
import json
import os
import time

import pika

# Configuration
DATASET_PATH = '/app/cub_200_2011/images'
RABBITMQ_QUEUE = 'raw_image_queue'


def connect_to_rabbitmq() -> pika.BlockingConnection:
    """
    Connect to RabbitMQ server with retry on failure.

    Returns:
        pika.BlockingConnection: RabbitMQ connection object.
    """
    while True:
        try:
            connection = pika.BlockingConnection(pika.ConnectionParameters('rabbitmq'))
            print("Connected to RabbitMQ.")
            return connection
        except pika.exceptions.AMQPConnectionError:
            print("RabbitMQ not available, retrying in 5 seconds...")
            time.sleep(5)


def publish_images_to_queue(dataset_path: str, queue_name: str) -> None:
    """
    Publish image data from a dataset directory to the RabbitMQ queue.

    Args:
        dataset_path (str): Path to the dataset directory.
        queue_name (str): Name of the RabbitMQ queue.
    """
    connection = connect_to_rabbitmq()
    channel = connection.channel()
    channel.queue_declare(queue=queue_name)

    for subdir, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.jpg'):
                image_path = os.path.join(subdir, file)
                try:
                    with open(image_path, 'rb') as image_file:
                        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

                    message = json.dumps({
                        'image_path': image_path,
                        'image_data': encoded_image
                    })

                    channel.basic_publish(exchange='', routing_key=queue_name, body=message)
                    print(f"Published {image_path} to {queue_name}")

                except Exception as e:
                    print(f"Failed to process {image_path}. Error: {e}")

    connection.close()
    print("Finished publishing images.")


if __name__ == "__main__":
    publish_images_to_queue(DATASET_PATH, RABBITMQ_QUEUE)
