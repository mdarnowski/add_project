import base64
import json
import logging
import threading
import time

import gridfs
import pika
import uvicorn
from bson import ObjectId
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pika.exceptions import AMQPConnectionError
from pymongo import MongoClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MongoDB configuration
MONGO_URI = "mongodb://mongodb:27017/"
client = MongoClient(MONGO_URI)
db = client.bird_dataset
images_collection = db.images
fs = gridfs.GridFS(db)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to store the progress
progress = {"processed": 0, "total": 0}


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
        except AMQPConnectionError:
            print("RabbitMQ not available, retrying in 5 seconds...")
            time.sleep(5)


def get_image(image_id):
    if not image_id:
        return None
    image_data = fs.get(image_id).read()
    return base64.b64encode(image_data).decode("utf-8")


def consume_progress_updates():
    global progress
    connection = connect_to_rabbitmq()
    channel = connection.channel()
    channel.queue_declare(queue="progress_queue")

    def callback(ch, method, properties, body):
        progress_update = json.loads(body)
        progress["processed"] = progress_update["processed"]
        progress["total"] = progress_update["total"]

    channel.basic_consume(
        queue="progress_queue", on_message_callback=callback, auto_ack=True
    )
    logger.info("Started consuming progress updates...")
    channel.start_consuming()


# Start the consumer in a separate thread
threading.Thread(target=consume_progress_updates, daemon=True).start()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/search", response_class=HTMLResponse)
async def search(
    request: Request,
    image_type: str = Query(None),
    species: str = Query(None),
    set_type: str = Query(None),
):
    query = {}
    if image_type:
        query["image_type"] = image_type
    if species:
        query["species"] = species
    if set_type:
        query["set_type"] = set_type

    logger.info(f"Query: {query}")

    images_cursor = images_collection.find(query).limit(100)
    images = []
    for data in images_cursor:
        image_id = data.get("image_id")
        image_data = {
            "species": data.get("species"),
            "set_type": data.get("set_type"),
            "image_path": data.get("filename"),
            "image_type": data.get("image_type"),
            "image": get_image(image_id),
        }
        images.append(image_data)

    logger.info(f"Images: {images}")
    return templates.TemplateResponse(
        "results.html", {"request": request, "images": images}
    )


@app.get("/progress")
def get_progress():
    return progress


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
