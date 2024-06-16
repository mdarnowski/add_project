# Use the official TensorFlow GPU base image
FROM tensorflow/tensorflow:latest-gpu

# Install MongoDB client and other dependencies
RUN apt-get update && \
    apt-get install -y python3-pip libgl1-mesa-glx

# Copy requirements.txt and install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Set the working directory
WORKDIR /app

# Copy your application files to the container
COPY trainer.py /app/
COPY rabbit_callback.py /app/

# Set the environment variable to suppress TensorFlow warnings
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV PYTHONUNBUFFERED=1
# Run the Python script
CMD ["python3", "trainer.py"]
