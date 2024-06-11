import pika
import json
import os

# RabbitMQ connection
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

channel.queue_declare(queue='data_queue')

dataset_path = 'cub_200_2011/images'  # Adjust to your path

for subdir, _, files in os.walk(dataset_path):
    for file in files:
        if file.endswith('.jpg'):
            image_path = os.path.join(subdir, file)
            message = json.dumps({'image_path': image_path})
            channel.basic_publish(exchange='', routing_key='data_queue', body=message)

connection.close()
