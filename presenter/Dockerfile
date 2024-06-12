FROM python:3.9-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY app.py /app/
COPY templates /app/templates

RUN pip install fastapi uvicorn pymongo jinja2

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
