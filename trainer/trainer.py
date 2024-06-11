import pika
import json
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
import numpy as np

# RabbitMQ connection
connection = pika.BlockingConnection(pika.ConnectionParameters('rabbitmq'))
channel = connection.channel()

channel.queue_declare(queue='train_data_queue')
channel.queue_declare(queue='val_data_queue')
channel.queue_declare(queue='test_data_queue')


# Define trainer architecture
def create_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(200, activation='softmax')(x)  # 200 classes

    model = Model(inputs=base_model.input, outputs=predictions)
    return model


model = create_model()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


def get_data(queue_name):
    dataset = []
    method_frame, header_frame, body = channel.basic_get(queue=queue_name, auto_ack=True)
    while method_frame:
        message = json.loads(body)
        image_data = np.array(message['data'])
        label = int(message['label'])
        dataset.append((image_data, label))
        method_frame, header_frame, body = channel.basic_get(queue=queue_name, auto_ack=True)
    return dataset


train_data = get_data('train_data_queue')
val_data = get_data('val_data_queue')

X_train, y_train = zip(*train_data)
X_val, y_val = zip(*val_data)

X_train = np.array(X_train)
y_train = np.array(y_train)
X_val = np.array(X_val)
y_val = np.array(y_val)

# Training
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Publish trained trainer weights to RabbitMQ
model_weights = model.get_weights()
channel.queue_declare(queue='trained_model')
channel.basic_publish(exchange='', routing_key='trained_model', body=json.dumps({'weights': model_weights}))

connection.close()
