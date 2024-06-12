FROM my-python-base

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY predictor.py /app/

CMD ["python", "predictor.py"]
