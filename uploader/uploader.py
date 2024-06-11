import json

import pika
from pymongo import MongoClient

client = MongoClient("mongodb://mongodb:27017/")
db = client.bird_dataset
collection = db.images

connection = pika.BlockingConnection(pika.ConnectionParameters("rabbitmq"))
channel = connection.channel()

channel.queue_declare(queue="processed_data_queue")


def callback(ch, method, properties, body):
    message = json.loads(body)
    collection.insert_one(message)


channel.basic_consume(
    queue="processed_data_queue", on_message_callback=callback, auto_ack=True
)
channel.start_consuming()
