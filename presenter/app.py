from flask import Flask, request, render_template, jsonify, redirect, url_for
import base64
import io
import json
import time

import numpy as np
import pika
from PIL import Image


# Connect to RabbitMQ
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


# Initialize Flask app
app = Flask(__name__)

# Establish RabbitMQ connection and channel
connection = connect_to_rabbitmq()
channel = connection.channel()

# Declare queues
channel.queue_declare(queue='prediction_queue')
channel.queue_declare(queue='prediction_response_queue')
channel.queue_declare(queue='train_request')


@app.route('/')
def index():
    """
    Render the index page.

    Returns:
        str: Rendered HTML template for the index page.
    """
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle image prediction requests.

    Returns:
        json: JSON response containing the prediction result.
    """
    file = request.files['file']
    image = Image.open(file.stream).resize((224, 224))

    # Encode image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Publish prediction request
    prediction_request = {
        'image_path': 'uploaded_image',
        'data': encoded_image
    }
    channel.basic_publish(exchange='', routing_key='prediction_queue', body=json.dumps(prediction_request))

    # Wait for prediction response
    method_frame, header_frame, body = channel.basic_get(queue='prediction_response_queue', auto_ack=True)
    while not method_frame:
        time.sleep(1)
        method_frame, header_frame, body = channel.basic_get(queue='prediction_response_queue', auto_ack=True)

    prediction_result = json.loads(body)

    return jsonify(prediction_result)


@app.route('/train', methods=['POST'])
def train():
    """
    Handle training requests.

    Returns:
        Response: Redirect to index page.
    """
    # Send message to trigger training
    channel.basic_publish(exchange='', routing_key='train_request', body=json.dumps({'train': True}))
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
