FROM python:3.9-slim

WORKDIR /app

COPY uploader.py /app/

RUN pip install pika pymongo

CMD ["python", "uploader.py"]
