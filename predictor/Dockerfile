FROM python:3.9-slim

WORKDIR /app

COPY predictor.py /app/

RUN pip install tensorflow pika numpy pillow pymongo

CMD ["python", "predictor.py"]
