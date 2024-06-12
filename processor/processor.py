import base64
import io
import json
import time

import pika
from loguru import logger
from PIL import Image


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


def send_to_queue(data: dict) -> None:
    """
    Send processed image data to the RabbitMQ queue.
    """
    connection = connect_to_rabbitmq()
    channel = connection.channel()
    channel.queue_declare(queue="processed_image_queue")

    try:
        image_path = data["image_path"]
        image_data = base64.b64decode(data["image_data"])

        # Process image
        image = Image.open(io.BytesIO(image_data)).resize((224, 224))
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="JPEG")

        label = image_path.split("/")[-2]

        processed_message = {
            "image_path": image_path,
            "data": base64.b64encode(image_bytes.getvalue()).decode("utf-8"),
            "label": label,
            "split": data["split"],
        }

        channel.basic_publish(
            exchange="",
            routing_key="processed_image_queue",
            body=json.dumps(processed_message),
        )
        logger.info(f"Sent processed message for image_path: {image_path} to queue")

    except Exception as e:
        logger.error(f"Error processing image_path: {image_path}. Error: {e}")

    connection.close()


# RabbitMQ connection for consuming
connection = connect_to_rabbitmq()
channel = connection.channel()

channel.queue_declare(queue="raw_image_queue")


def callback(ch, method, properties, body) -> None:
    """
    Callback function to handle incoming messages from the RabbitMQ queue.
    """
    message = json.loads(body)
    send_to_queue(message)
    logger.info(f"Processed and sent message for image_path: {message['image_path']}")


channel.basic_consume(
    queue="raw_image_queue", on_message_callback=callback, auto_ack=True
)

logger.info("Processor is listening for messages...")
channel.start_consuming()
