from pymongo import MongoClient
import base64
import json
import pika
import os
import gridfs
from loguru import logger
import tensorflow as tf
import time

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
MODEL_CHUNKS_QUEUE = os.getenv("MODEL_CHUNKS_QUEUE", "model_chunks_queue")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongodb:27017/")
MONGO_DB = os.getenv("MONGO_DB", "model_db")

client = MongoClient(MONGO_URI)
db = client[MONGO_DB]
fs = gridfs.GridFS(db)

class ModelUploader:
    def __init__(self):
        self.connection = None
        self.channel = None
        self.chunks = {}
        self.connect_to_rabbitmq()

    def connect_to_rabbitmq(self):
        while True:
            try:
                self.connection = pika.BlockingConnection(
                    pika.ConnectionParameters(host=RABBITMQ_HOST)
                )
                self.channel = self.connection.channel()
                self.channel.queue_declare(queue=MODEL_CHUNKS_QUEUE)
                self.channel.basic_consume(
                    queue=MODEL_CHUNKS_QUEUE,
                    on_message_callback=self.receive_chunk,
                    auto_ack=True,
                )
                logger.info("Connected to RabbitMQ.")
                break
            except pika.exceptions.AMQPConnectionError as e:
                logger.warning(f"Connection to RabbitMQ failed: {e}. Retrying in 5 seconds...")
                time.sleep(5)

    def receive_chunk(self, ch, method, properties, body):
        try:
            message = json.loads(body)
            chunk_index = message["chunk_index"]
            total_chunks = message["total_chunks"]
            data = base64.b64decode(message["data"])

            if chunk_index not in self.chunks:
                self.chunks[chunk_index] = data

            if len(self.chunks) == total_chunks:
                self.save_model()
                self.chunks.clear()
        except Exception as e:
            logger.error(f"Error processing chunk: {e}")

    def save_model(self):
        try:
            ordered_chunks = [self.chunks[i] for i in sorted(self.chunks.keys())]
            model_data = b"".join(ordered_chunks)

            with open("received_model.keras", "wb") as f:
                f.write(model_data)

            tf.keras.models.load_model("received_model.keras")
            logger.info("Model loaded successfully from received data.")

            fs.put(model_data, filename="model.keras")
            logger.info("Model saved to MongoDB GridFS")
        except Exception as e:
            logger.error(f"Error saving model: {e}")

    def start_consuming(self):
        while True:
            try:
                logger.info("Starting consumption.")
                self.channel.start_consuming()
            except pika.exceptions.StreamLostError as e:
                logger.warning(f"Stream connection lost: {e}. Attempting to reconnect...")
                self.connect_to_rabbitmq()
            except Exception as e:
                logger.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    uploader = ModelUploader()
    uploader.start_consuming()
