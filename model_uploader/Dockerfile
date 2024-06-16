FROM my-python-base

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model_uploader.py /app/

CMD ["python", "model_uploader.py"]
