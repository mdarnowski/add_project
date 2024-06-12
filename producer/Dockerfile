FROM python:3.9-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY producer.py /app/

RUN pip install pika loguru

CMD ["python", "producer.py"]
