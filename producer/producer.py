import json
import os
import random

import pika

connection = pika.BlockingConnection(pika.ConnectionParameters("rabbitmq"))
channel = connection.channel()

channel.queue_declare(queue="train_data_queue")
channel.queue_declare(queue="val_data_queue")
channel.queue_declare(queue="test_data_queue")

dataset_path = "/app/cub_200_2011/images"
data = []

for subdir, _, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".jpg"):
            image_path = os.path.join(subdir, file)
            data.append(image_path)

random.shuffle(data)
total_data = len(data)
train_split = int(total_data * 0.7)
val_split = int(total_data * 0.1)
test_split = total_data - train_split - val_split

train_data = data[:train_split]
val_data = data[train_split : train_split + val_split]
test_data = data[train_split + val_split :]


def send_data(queue_name, dataset):
    for image_path in dataset:
        message = json.dumps({"image_path": image_path})
        channel.basic_publish(exchange="", routing_key=queue_name, body=message)


send_data("train_data_queue", train_data)
send_data("val_data_queue", val_data)
send_data("test_data_queue", test_data)

connection.close()
