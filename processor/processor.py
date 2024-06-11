import io
import json

import numpy as np
import pika
from PIL import Image


def process_image(image_path):
    with Image.open(image_path) as img:
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
    return img_array


# RabbitMQ connection
connection = pika.BlockingConnection(pika.ConnectionParameters("localhost"))
channel = connection.channel()

channel.queue_declare(queue="processed_data_queue")


def callback(ch, method, properties, body):
    message = json.loads(body)
    image_path = message["image_path"]
    processed_data = process_image(image_path)
    processed_message = json.dumps(
        {"image_path": image_path, "data": processed_data.tolist()}
    )
    channel.basic_publish(
        exchange="", routing_key="processed_data_queue", body=processed_message
    )


channel.basic_consume(queue="data_queue", on_message_callback=callback, auto_ack=True)
channel.start_consuming()
