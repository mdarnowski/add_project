import base64
import io
import json
import time

import pika
import torchvision.transforms as transforms
from loguru import logger
from PIL import Image


def connect_to_rabbitmq() -> pika.BlockingConnection:
    while True:
        try:
            conn = pika.BlockingConnection(pika.ConnectionParameters("rabbitmq"))
            logger.info("Connected to RabbitMQ.")
            return conn
        except pika.exceptions.AMQPConnectionError:
            logger.warning("RabbitMQ not available, retrying in 5 seconds...")
            time.sleep(5)


def send_to_queue(data: dict) -> None:
    conn = connect_to_rabbitmq()
    channel = conn.channel()
    channel.queue_declare(queue="processed_image_queue")

    try:
        image_path = data["image_path"]
        image_data = base64.b64decode(data["image_data"])

        # Load image
        image = Image.open(io.BytesIO(image_data))

        # Define transformations
        transform = transforms.Compose(
            [
                transforms.Resize(
                    (256, 256)
                ),  # Resize to larger, then crop to desired size
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomRotation(10),  # Slightly reduced rotation
                transforms.RandomAffine(
                    degrees=0, translate=(0.1, 0.1)
                ),  # Slight translations
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # ImageNet mean and std
            ]
        )

        # Apply transformations
        transformed_image = transform(image)

        # Convert tensor to PIL Image for saving
        transformed_image = transforms.ToPILImage()(transformed_image)

        # Save transformed image
        image_bytes = io.BytesIO()
        transformed_image.save(image_bytes, format="JPEG")

        processed_message = {
            "image_path": image_path,
            "processed_image_data": base64.b64encode(image_bytes.getvalue()).decode(
                "utf-8"
            ),
            "label": data["label"],
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

    conn.close()


connection = connect_to_rabbitmq()
channel = connection.channel()

channel.queue_declare(queue="raw_image_queue")


def callback(ch, method, properties, body) -> None:
    message = json.loads(body)
    send_to_queue(message)
    logger.info(f"Processed and sent message for image_path: {message['image_path']}")


channel.basic_consume(
    queue="raw_image_queue", on_message_callback=callback, auto_ack=True
)

logger.info("Processor is listening for messages...")
channel.start_consuming()
