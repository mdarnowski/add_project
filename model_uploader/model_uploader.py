from pymongo import MongoClient
import base64
import json
import pika
import os
import gridfs
from loguru import logger
import time
import threading

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
        self.total_chunks = 0
        self.all_chunks_received = threading.Event()
        self.connect_to_rabbitmq()
        self.lock = threading.Lock()

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
            except pika.exceptions.AMQPConnectionError:
                logger.warning("RabbitMQ not available, retrying in 5 seconds...")
                time.sleep(5)

    def receive_chunk(self, ch, method, properties, body):
        try:
            message = json.loads(body)
            chunk_index = message["chunk_index"]
            print(f"Received chunk: {chunk_index}")
            self.total_chunks = message["total_chunks"]
            print(f"Total chunks: {self.total_chunks}")
            data = base64.b64decode(message["data"])

            with self.lock:  # Ensure thread-safe access to self.chunks
                if chunk_index not in self.chunks:
                    self.chunks[chunk_index] = data
                print(f"Chunks received: {len(self.chunks)}")

            if len(self.chunks) == self.total_chunks:
                self.all_chunks_received.set()  # Signal that all chunks are received
        except Exception as e:
            logger.error(f"Error processing chunk: {e}")

    def save_model(self):
        try:
            with self.lock:  # Ensure thread-safe access to self.chunks
                ordered_chunks = [self.chunks[i] for i in sorted(self.chunks.keys())]
                model_data = b"".join(ordered_chunks)

            # Writing to file to verify model integrity
            with open("received_model.keras", "wb") as f:
                f.write(model_data)

            # Attempt to load the model to check if it's valid
            #tf.keras.models.load_model("received_model.keras")
            #logger.info("Model loaded successfully from received data.")

            # Save model to MongoDB GridFS
            fs.put(model_data, filename="model.keras")
            logger.info("Model saved to MongoDB GridFS")

        except Exception as e:
            logger.error(f"Error during save_model: {e}")

    def start_consuming(self):
        threading.Thread(target=self._consume).start()
        threading.Thread(target=self._monitor_chunks).start()

    def _consume(self):
        while True:
            try:
                self.channel.start_consuming()
            except pika.exceptions.StreamLostError:
                logger.warning("Stream connection lost, attempting to reconnect...")
                self.connect_to_rabbitmq()

    def _monitor_chunks(self):
        while True:
            self.all_chunks_received.wait()  # Wait until all chunks are received
            logger.info("All chunks received, starting to save the model.")
            self.save_model()
            with self.lock:  # Ensure thread-safe modification of self.chunks
                self.chunks.clear()  # Clear chunks after saving the model
            self.all_chunks_received.clear()  # Reset the event
            logger.info("Model saved and chunks cleared.")


if __name__ == "__main__":
    uploader = ModelUploader()
    uploader.start_consuming()
