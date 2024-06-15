"""
Web Application Module
================================

This module provides a web application for managing and processing images using FastAPI.
It integrates with RabbitMQ for messaging and MongoDB/GridFS for image storage.

Modules
-------

- `base64`
- `json`
- `logging`
- `threading`
- `time`
- `gridfs`
- `pika`
- `uvicorn`
- `fastapi`
- `fastapi.middleware.cors`
- `fastapi.responses`
- `fastapi.templating`
- `pika.exceptions`
- `pymongo`

Attributes
----------

- `MONGO_URI`: MongoDB connection URI.
- `client`: MongoDB client.
- `db`: MongoDB database.
- `images_collection`: MongoDB collection for images.
- `fs`: GridFS object for storing images.
- `app`: FastAPI application instance.
- `templates`: Jinja2Templates object for HTML rendering.
- `progress`: Global dictionary to store processing progress.

Functions
---------

- `connect_to_rabbitmq()`: Connect to RabbitMQ server with retry on failure.
- `get_image(image_id)`: Retrieve image from GridFS and encode it in base64.
- `consume_progress_updates()`: Consume progress updates from RabbitMQ.
- `index(request: Request)`: Serve the main index page.
- `search(request: Request, image_type: str, species: str, set_type: str)`: Serve the search results page.
- `get_progress()`: Return current processing progress.
- `trigger_processing()`: Trigger image processing via RabbitMQ.
"""

import base64
import json
import logging
import threading
import time
import asyncio

import gridfs
import pika
import uvicorn
from fastapi import FastAPI, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pika.exceptions import AMQPConnectionError
from pymongo import MongoClient
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MongoDB configuration
MONGO_URI = "mongodb://mongodb:27017/"
client = MongoClient(MONGO_URI)
db = client.bird_dataset
images_collection = db.images
metrics_collection = db.metrics
fs = gridfs.GridFS(db)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
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
progress_producer = {
    "produced": 0,
    "processed": 0,
    "uploaded_raw": 0,
    "uploaded_processed": 0,
    "total": 0,
}


# WebSocket manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


manager = ConnectionManager()


def connect_to_rabbitmq() -> pika.BlockingConnection:
    """
    Connect to RabbitMQ server with retry on failure.

    Returns
    -------
    pika.BlockingConnection
        RabbitMQ connection object.
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
    """
    Retrieve image from GridFS and encode it in base64.

    Parameters
    ----------
    image_id : ObjectId
        The GridFS ObjectId of the image.

    Returns
    -------
    str
        Base64 encoded image data.
    """
    if not image_id:
        return None
    image_data = fs.get(image_id).read()
    return base64.b64encode(image_data).decode("utf-8")


def consume_updates():
    global progress_producer
    connection = connect_to_rabbitmq()
    channel = connection.channel()
    channel.queue_declare(queue="producer_progress_queue")
    channel.queue_declare(queue="processor_progress_queue")
    channel.queue_declare(queue="raw_uploader_progress_queue")
    channel.queue_declare(queue="processed_uploader_progress_queue")
    channel.queue_declare(queue="training_updates")

    def callback(ch, method, properties, body):
        update = json.loads(body)
        asyncio.run(manager.broadcast(json.dumps(update)))

    def callback_producer(ch, method, properties, body):
        progress_update = json.loads(body)
        progress_producer["produced"] = progress_update["produced"]
        progress_producer["total"] = progress_update["total"]

    def callback_processor(ch, method, properties, body):
        progress_update = json.loads(body)
        progress_producer["processed"] = progress_update["processed"]

    def callback_raw_uploader(ch, method, properties, body):
        progress_update = json.loads(body)
        progress_producer["uploaded_raw"] = progress_update["uploaded"]

    def callback_processed_uploader(ch, method, properties, body):
        progress_update = json.loads(body)
        progress_producer["uploaded_processed"] = progress_update["uploaded"]

    channel.basic_consume(
        queue="producer_progress_queue",
        on_message_callback=callback_producer,
        auto_ack=True,
    )
    channel.basic_consume(
        queue="processor_progress_queue",
        on_message_callback=callback_processor,
        auto_ack=True,
    )
    channel.basic_consume(
        queue="raw_uploader_progress_queue",
        on_message_callback=callback_raw_uploader,
        auto_ack=True,
    )
    channel.basic_consume(
        queue="processed_uploader_progress_queue",
        on_message_callback=callback_processed_uploader,
        auto_ack=True,
    )
    channel.basic_consume(
        queue="training_updates", on_message_callback=callback, auto_ack=True
    )
    logger.info("Started consuming updates...")
    channel.start_consuming()


# Start the consumer in a separate thread
threading.Thread(target=consume_updates, daemon=True).start()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    Serve the main index page.

    Parameters
    ----------
    request : Request
        FastAPI request object.

    Returns
    -------
    HTMLResponse
        Rendered HTML page with species list.
    """
    species_list = images_collection.distinct("species")
    return templates.TemplateResponse(
        "index.html", {"request": request, "species_list": species_list}
    )


@app.get("/search", response_class=HTMLResponse)
async def search(
    request: Request,
    image_type: str = Query(None),
    species: str = Query(None),
    set_type: str = Query(None),
):
    """
    Serve the search results page based on the query parameters.

    Parameters
    ----------
    request : Request
        FastAPI request object.
    image_type : str, optional
        Type of the image (e.g., "jpg", "png").
    species : str, optional
        Species of the bird.
    set_type : str, optional
        Set type (e.g., "train", "val").

    Returns
    -------
    HTMLResponse
        Rendered HTML page with search results.
    """
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
    """
    Return current processing progress.

    Returns
    -------
    dict
        Dictionary containing the processed and total counts for all stages.
    """
    return progress_producer


@app.get("/latest_metrics")
def get_latest_metrics():
    latest_training = metrics_collection.find_one(sort=[("timestamp", -1)])
    if latest_training:
        return {"epochs": latest_training["epochs"]}
    else:
        return {"epochs": []}


@app.post("/start_training")
def start_training():
    """
    Start training via RabbitMQ.

    Returns
    -------
    dict
        Confirmation message for starting training.
    """
    images_in_db_count = images_collection.count_documents({"image_type": "processed"})
    if progress_producer["produced"] == 0:
        return {
            "message": "Producer has not started producing images, please start producing first."
        }
    elif images_in_db_count >= progress_producer["total"]:
        connection = connect_to_rabbitmq()
        channel = connection.channel()
        channel.queue_declare(queue="training_queue")
        channel.basic_publish(exchange="", routing_key="training_queue", body="")
        connection.close()
        return {"message": "All images processed, starting training..."}
    else:
        return {
            "message": f"Not all images processed, cannot start training (images: {images_in_db_count} / {progress_producer['total']})"
        }


@app.post("/trigger_processing")
def trigger_processing():
    """
    Trigger image processing via RabbitMQ.

    Returns
    -------
    dict
        Confirmation message for triggering image processing.
    """
    connection = connect_to_rabbitmq()
    channel = connection.channel()
    channel.queue_declare(queue="trigger_queue")
    channel.basic_publish(exchange="", routing_key="trigger_queue", body="")
    connection.close()
    return {"message": "Image processing triggered"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.send_personal_message(f"You wrote: {data}", websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
