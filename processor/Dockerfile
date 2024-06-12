FROM python:3.9-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY processor.py /app/

RUN pip install pika pillow numpy loguru torchvision

CMD ["python", "processor.py"]
