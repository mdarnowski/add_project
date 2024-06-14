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


def consume_updates():
    connection = connect_to_rabbitmq()
    channel = connection.channel()
    channel.queue_declare(queue="progress_queue")
    channel.queue_declare(queue="training_updates")

    def callback(ch, method, properties, body):
        update = json.loads(body)
        asyncio.run(manager.broadcast(json.dumps(update)))

    channel.basic_consume(
        queue="progress_queue", on_message_callback=callback, auto_ack=True
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


@app.get("/latest_metrics")
def get_latest_metrics():
    latest_training = metrics_collection.find_one(sort=[("timestamp", -1)])
    if latest_training:
        return {"epochs": latest_training["epochs"]}
    else:
        return {"epochs": []}


@app.post("/trigger_processing")
def trigger_processing():
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
