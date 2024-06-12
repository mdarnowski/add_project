import base64
import io
import json
import time

import numpy as np
import pika
from flask import Flask, jsonify, redirect, render_template, request, url_for
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
            connection = pika.BlockingConnection(pika.ConnectionParameters("rabbitmq"))
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


@app.route("/")
def index():
    """
    Render the index page.

    Returns:
        str: Rendered HTML template for the index page.
    """
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
