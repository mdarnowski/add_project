import os
import io
import json
import base64
import numpy as np
import tensorflow as tf
from PIL import Image
from pymongo import MongoClient
import pika
import gridfs
import threading
import time

# Configuration constants
MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongodb:27017/")
MONGO_DB = os.getenv("MONGO_DB", "model_db")
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
TO_PREDICT_QUEUE = os.getenv("TO_PREDICT_QUEUE", "to_predict_queue")
PREDICTION_QUEUE = os.getenv("PREDICTION_QUEUE", "prediction_queue")

# MongoDB setup
client = MongoClient(MONGO_URI)
db = client[MONGO_DB]
fs = gridfs.GridFS(db)

CHECK_INTERVAL = 3  # Check for a new model every 3 seconds


class Predictor:
    def __init__(self):
        self.model = None
        self.current_model_upload_date = None
        self.connection = None
        self.channel = None
        self.load_initial_model()
        self.connect_to_rabbitmq()
        self.start_model_check_thread()

    def load_initial_model(self):
        self.model, self.current_model_upload_date = self.download_model()

    def download_model(self):
        try:
            latest_file = fs.find_one(sort=[("uploadDate", -1)])
            if latest_file and (
                self.current_model_upload_date is None
                or latest_file.uploadDate > self.current_model_upload_date
            ):
                model_data = latest_file.read()
                with open("downloaded_model.keras", "wb") as f:
                    f.write(model_data)
                model = tf.keras.models.load_model("downloaded_model.keras")
                print("Downloaded and loaded new model.")
                return model, latest_file.uploadDate
            print("No new model found in GridFS.")
            return self.model, self.current_model_upload_date
        except Exception as e:
            print(f"Error downloading model: {e}")
            return self.model, self.current_model_upload_date

    def check_and_update_model(self):
        while True:
            try:
                print("Checking for a new model...")
                new_model, new_upload_date = self.download_model()
                if new_model and (
                    self.current_model_upload_date is None
                    or new_upload_date > self.current_model_upload_date
                ):
                    self.model = new_model
                    self.current_model_upload_date = new_upload_date
                    print("Model updated.")
            except Exception as e:
                print(f"Error during model check: {e}")
            time.sleep(CHECK_INTERVAL)

    def start_model_check_thread(self):
        thread = threading.Thread(target=self.check_and_update_model, daemon=True)
        thread.start()

    @staticmethod
    def preprocess_image(image_data):
        try:
            image = Image.open(io.BytesIO(image_data))
            image = image.resize((299, 299))
            image_array = np.array(image)
            return tf.cast(image_array, tf.float32) / 255.0
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None

    def predict(self, image_data):
        if self.model is None:
            return -1
        image_array = self.preprocess_image(image_data)
        if image_array is None:
            return -1
        predictions = self.model.predict(image_array)
        predicted_label = np.argmax(predictions, axis=1)
        return int(predicted_label[0])

    def connect_to_rabbitmq(self):
        while True:
            try:
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
                print("Connected to RabbitMQ.")
                break
            except pika.exceptions.AMQPConnectionError:
                print("RabbitMQ not available, retrying in 5 seconds...")
                time.sleep(5)

    def on_request(self, _ch, _method, _properties, body):
        try:
            message = json.loads(body)
            image_data = base64.b64decode(message["image_data"])
            predicted_label = self.predict(image_data)
            response = {
                "image_path": message["image_path"],
                "image_type": message["image_type"],
                "predicted_label": predicted_label,
            }
            self.channel.basic_publish(
                exchange="",
                routing_key=PREDICTION_QUEUE,
                body=json.dumps(response),
            )
        except Exception as e:
            print(f"Error processing request: {e}")

    def start_consuming(self):
        print("Starting to consume messages.")
        self.channel.start_consuming()


if __name__ == "__main__":
    predictor = Predictor()
    predictor.start_consuming()
