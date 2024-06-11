import io
import json

import numpy as np
import pika
from flask import Flask, jsonify, render_template, request
from PIL import Image

app = Flask(__name__)

# RabbitMQ connection
connection = pika.BlockingConnection(pika.ConnectionParameters("rabbitmq"))
channel = connection.channel()

channel.queue_declare(queue="prediction_queue")
channel.queue_declare(queue="prediction_response_queue")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    image = Image.open(file.stream).resize((224, 224))
    image_array = np.array(image) / 255.0

    # Send image data to prediction queue
    prediction_request = {"image_path": "uploaded_image", "data": image_array.tolist()}
    channel.basic_publish(
        exchange="", routing_key="prediction_queue", body=json.dumps(prediction_request)
    )

    # Get prediction response
    method_frame, header_frame, body = channel.basic_get(
        queue="prediction_response_queue", auto_ack=True
    )
    while not method_frame:
        method_frame, header_frame, body = channel.basic_get(
            queue="prediction_response_queue", auto_ack=True
        )

    prediction_result = json.loads(body)

    return jsonify(prediction_result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
