FROM python:3.9-slim

WORKDIR /app

COPY app.py /app/
COPY templates /app/templates
COPY ../model/model.h5 /app/model.h5

RUN pip install flask tensorflow pillow numpy

CMD ["python", "app.py"]
