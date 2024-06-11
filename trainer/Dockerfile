FROM python:3.9-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY trainer.py /app/

RUN pip install tensorflow pika pillow pymongo numpy

CMD ["python", "trainer.py"]
