import base64
import logging

import gridfs
import uvicorn
from bson import ObjectId
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
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


def get_image(image_id):
    if not image_id:
        return None
    image_data = fs.get(image_id).read()
    return base64.b64encode(image_data).decode("utf-8")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    species_list = images_collection.distinct("species")
    return templates.TemplateResponse("index.html", {"request": request, "species_list": species_list})


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


if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8000)
