import json
import time

import pika
from pymongo import MongoClient

# MongoDB connection
MONGO_URI = 'mongodb://mongodb:27017/'
client = MongoClient(MONGO_URI)
db = client.bird_dataset
collection = db.images


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


# RabbitMQ connection
connection = connect_to_rabbitmq()
channel = connection.channel()

channel.queue_declare(queue='processed_image_queue')


def callback(ch, method, properties, body) -> None:
    """
    Callback function to handle incoming messages from the RabbitMQ queue.

    Args:
        ch: Channel.
        method: Method.
        properties: Properties.
        body: Message body containing image data.
    """
    message = json.loads(body)
    try:
        # Extract and store only the necessary parts
        image_doc = {
            'image_path': message['image_path'],
            'data': message['data'],  # base64 encoded preprocessed image data
            'label': message['label'],
            'split': message['split']
        }
        collection.insert_one(image_doc)
        print(f"Inserted document with image_path: {message['image_path']}")
    except Exception as e:
        print(f"Error inserting document: {e}")


channel.basic_consume(queue='processed_image_queue', on_message_callback=callback, auto_ack=True)

print('Uploader is listening for messages...')
channel.start_consuming()
