FROM python:3.9-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY app.py /app/
COPY templates /app/templates

RUN pip install flask pika pillow numpy

CMD ["python", "app.py"]
