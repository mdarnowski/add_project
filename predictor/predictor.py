import json

import numpy as np
import pika
import tensorflow as tf
from tensorflow.keras.models import Model


# Define trainer architecture
def create_model():
    base_model = tf.keras.applications.VGG16(
        weights=None, include_top=False, input_shape=(224, 224, 3)
    )
    x = base_model.output
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    predictions = tf.keras.layers.Dense(200, activation="softmax")(x)  # 200 classes

    model = Model(inputs=base_model.input, outputs=predictions)
    return model


model = create_model()

# RabbitMQ connection
connection = pika.BlockingConnection(pika.ConnectionParameters("rabbitmq"))
channel = connection.channel()

channel.queue_declare(queue="trained_model")
channel.queue_declare(queue="prediction_queue")
channel.queue_declare(queue="prediction_response_queue")


# Load trainer weights
def load_model_weights(ch, method, properties, body):
    message = json.loads(body)
    model.set_weights(message["weights"])
    print("Model weights loaded")


channel.basic_consume(
    queue="trained_model", on_message_callback=load_model_weights, auto_ack=True
)


# Predict
def predict_image(image_array):
    predictions = model.predict(np.expand_dims(image_array, axis=0))
    return np.argmax(predictions, axis=1)[0]


def callback(ch, method, properties, body):
    message = json.loads(body)
    image_data = np.array(message["data"])
    prediction = predict_image(image_data)
    response = json.dumps(
        {"image_path": message["image_path"], "prediction": int(prediction)}
    )
    channel.basic_publish(
        exchange="", routing_key="prediction_response_queue", body=response
    )


channel.basic_consume(
    queue="prediction_queue", on_message_callback=callback, auto_ack=True
)
channel.start_consuming()
