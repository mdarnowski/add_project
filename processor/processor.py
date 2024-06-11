import base64
import io
import json
import random
import threading
import time
from PIL import Image

import pika

# Global variables
image_data_list = []
lock = threading.Lock()


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


def send_to_queue(data: list, split_type: str) -> None:
    """
    Send processed image data to the RabbitMQ queue.

    Args:
        data (list): List of image data to be processed.
        split_type (str): Data split type ('train', 'val', 'test').
    """
    connection = connect_to_rabbitmq()
    channel = connection.channel()
    channel.queue_declare(queue='processed_image_queue')

    for message in data:
        try:
            image_path = message['image_path']
            image_data = base64.b64decode(message['image_data'])

            # Process image
            image = Image.open(io.BytesIO(image_data)).resize((224, 224))
            image_bytes = io.BytesIO()
            image.save(image_bytes, format='JPEG')

            label = image_path.split('/')[-2]  # Assuming class names are directory names

            processed_message = {
                'image_path': image_path,
                'data': base64.b64encode(image_bytes.getvalue()).decode('utf-8'),
                'label': label,
                'split': split_type
            }

            channel.basic_publish(exchange='', routing_key='processed_image_queue', body=json.dumps(processed_message))
            print(f"Sent processed message for image_path: {image_path} to queue")

        except Exception as e:
            print(f"Error processing image_path: {image_path}. Error: {e}")
            break

        time.sleep(0.1)  # Throttle message sending to prevent overwhelming RabbitMQ

    connection.close()


def process_and_send_data() -> None:
    """
    Process and send image data to RabbitMQ queues for training, validation, and testing.
    """
    while True:
        time.sleep(60)  # Process every 60 seconds

        with lock:
            if not image_data_list:
                print("No data to process.")
                continue

            print(f"Processing {len(image_data_list)} items from raw_image_queue.")

            # Shuffle and split the dataset
            random.shuffle(image_data_list)
            total_data = len(image_data_list)
            train_split = int(total_data * 0.7)
            val_split = int(total_data * 0.1)
            test_split = total_data - train_split - val_split

            train_data = image_data_list[:train_split]
            val_data = image_data_list[train_split:train_split + val_split]
            test_data = image_data_list[train_split + val_split:]

            print("Sending train data...")
            send_to_queue(train_data, 'train')

            print("Sending validation data...")
            send_to_queue(val_data, 'val')

            print("Sending test data...")
            send_to_queue(test_data, 'test')

            image_data_list.clear()  # Clear the list after processing


# Start a separate thread for data processing
processing_thread = threading.Thread(target=process_and_send_data)
processing_thread.daemon = True
processing_thread.start()

# RabbitMQ connection for consuming
connection = connect_to_rabbitmq()
channel = connection.channel()

channel.queue_declare(queue='raw_image_queue')


def callback(ch, method, properties, body) -> None:
    """
    Callback function to handle incoming messages from the RabbitMQ queue.

    Args:
        ch: Channel.
        method: Method.
        properties: Properties.
        body: Message body.
    """
    message = json.loads(body)
    with lock:
        image_data_list.append(message)
    print(f"Received message for image_path: {message['image_path']}")


channel.basic_consume(queue='raw_image_queue', on_message_callback=callback, auto_ack=True)

print('Processor is listening for messages...')
channel.start_consuming()
