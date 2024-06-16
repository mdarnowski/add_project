import io
import numpy as np
import tensorflow as tf
from pymongo import MongoClient
import os
from PIL import Image
import pika
import base64
import json
import gridfs

MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongodb:27017/")
MONGO_DB = os.getenv("MONGO_DB", "model_db")
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
TO_PREDICT_QUEUE = os.getenv("TO_PREDICT_QUEUE", "to_predict_queue")
PREDICTION_QUEUE = os.getenv("PREDICTION_QUEUE", "prediction_queue")

client = MongoClient(MONGO_URI)
db = client[MONGO_DB]
fs = gridfs.GridFS(db)


class Predictor:
    def __init__(self):
        self.model = self.download_model()
        self.connection = None
        self.channel = None
        self.connect_to_rabbitmq()

    def download_model(self):
        latest_file = fs.find_one(
            sort=[("uploadDate", -1)]
        )  # Get the latest model file from GridFS
        if latest_file:
            model_data = latest_file.read()
            with open("downloaded_model.keras", "wb") as f:
                f.write(model_data)
            model = tf.keras.models.load_model("downloaded_model.keras")
            return model
        else:
            return None  # No model found

    def preprocess_image(self, image_data):
        image = Image.open(io.BytesIO(image_data))
        image = image.resize((299, 299))
        image_array = np.array(image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = tf.keras.applications.inception_v3.preprocess_input(image_array)
        return image_array

    def predict(self, image_data):
        if self.model is None:
            return -1
        image_array = self.preprocess_image(image_data)
        predictions = self.model.predict(image_array)
        predicted_label = np.argmax(predictions, axis=1)
        return predicted_label

    def connect_to_rabbitmq(self):
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=RABBITMQ_HOST)
        )
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue=TO_PREDICT_QUEUE)
        self.channel.queue_declare(queue=PREDICTION_QUEUE)
        self.channel.basic_consume(
            queue=TO_PREDICT_QUEUE,
            on_message_callback=self.on_request,
            auto_ack=True,
        )

    def on_request(self, _ch, _method, _properties, body):
        message = json.loads(body)
        image_data = base64.b64decode(message["image_data"])
        predicted_label = self.predict(image_data)
        response = {
            "predicted_label": int(predicted_label) if predicted_label != -1 else -1
        }
        self.channel.basic_publish(
            exchange="",
            routing_key=PREDICTION_QUEUE,
            body=json.dumps(response),
        )

    def start_consuming(self):
        self.channel.start_consuming()


if __name__ == "__main__":
    predictor = Predictor()
    predictor.start_consuming()
