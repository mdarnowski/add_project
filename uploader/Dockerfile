FROM python:3.9-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY uploader.py /app/

RUN pip install pika pymongo loguru

CMD ["python", "uploader.py"]
