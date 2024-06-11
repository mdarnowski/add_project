FROM python:3.9-slim

WORKDIR /app

COPY producer.py /app/
COPY cub_200_2011 /app/cub_200_2011

RUN pip install pika

CMD ["python", "producer.py"]
