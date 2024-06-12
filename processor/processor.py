import base64
import io
import json
import time

import pika
import torchvision.transforms as transforms
from loguru import logger
from pika.exceptions import AMQPConnectionError
from PIL import Image


class RabbitMQProcessor:
    def __init__(self):
        self.connection = None
        self.channel = None
        self.connect_to_rabbitmq()

    def connect_to_rabbitmq(self):
        while self.connection is None or self.connection.is_closed:
            try:
                self.connection = pika.BlockingConnection(
                    pika.ConnectionParameters("rabbitmq")
                )
                self.channel = self.connection.channel()
                self.channel.queue_declare(queue="processed_image_queue")
                logger.info("Connected to RabbitMQ.")
            except AMQPConnectionError:
                logger.warning("RabbitMQ not available, retrying in 5 seconds...")
                time.sleep(5)

    def send_to_queue(self, data: dict):
        try:
            if self.channel is None or self.channel.is_closed:
                self.connect_to_rabbitmq()

            image_path = data["image_path"]
            image_data = base64.b64decode(data["image_data"])

            # Load image
            image = Image.open(io.BytesIO(image_data))

            if data["split"] == "train":
                transform = transforms.Compose(
                    [
                        transforms.Resize([224, 224]),
                        transforms.ColorJitter(
                            brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
                        ),
                        transforms.ToTensor(),
                    ]
                )
            else:  # val or test
                transform = transforms.Compose(
                    [
                        transforms.Resize([224, 224]),
                        transforms.ToTensor(),
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
                "species": data["species"],
            }

            self.channel.basic_publish(
                exchange="",
                routing_key="processed_image_queue",
                body=json.dumps(processed_message),
            )
            logger.info(f"Sent processed message for image_path: {image_path} to queue")

        except Exception as e:
            logger.error(f"Error processing image. Error: {e}")
            if isinstance(e, AMQPConnectionError):
                logger.error("Connection lost, attempting to reconnect...")
                self.connect_to_rabbitmq()

    def start_consuming(self):
        if self.channel is None or self.channel.is_closed:
            self.connect_to_rabbitmq()

        self.channel.queue_declare(queue="raw_image_queue")

        def callback(ch, method, properties, body):
            message = json.loads(body)
            self.send_to_queue(message)
            logger.info(
                f"Processed and sent message for image_path: {message['image_path']}"
            )

        self.channel.basic_consume(
            queue="raw_image_queue", on_message_callback=callback, auto_ack=True
        )
        logger.info("Processor is listening for messages...")
        self.channel.start_consuming()


if __name__ == "__main__":
    processor = RabbitMQProcessor()
    processor.start_consuming()
