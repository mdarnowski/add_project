FROM python:3.9-slim

WORKDIR /app

COPY processor.py /app/

RUN pip install pika pillow numpy

CMD ["python", "processor.py"]
