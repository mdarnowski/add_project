"""
Trainer Module
==============

This module provides functions for training a deep learning model using TensorFlow and Keras.
It handles data retrieval from MongoDB, preparation of datasets in TFRecord format, model creation,
training, and evaluation.

Functions
---------
- `create_model(num_classes: int) -> tf.keras.models.Model`
    Creates and compiles a Keras model using InceptionV3 as the base model.
- `parse_tfrecord(tfrecord)`
    Parses a single TFRecord example into an image tensor and label.
- `load_tfrecord_dataset(tfrecord_path: str, batch_size: int, shuffle_buffer_size: int, num_classes: int)`
    Loads a TFRecord dataset and prepares it for training.
- `download_data_and_save_to_tfrecord(split: str, tfrecord_path: str)`
    Downloads data from MongoDB and saves it to a TFRecord file.
- `train_and_evaluate_model() -> None`
    Trains and evaluates the model using the downloaded datasets.

Environment Variables
---------------------
- `MONGO_URI` : str
    MongoDB connection URI, defaulting to "mongodb://mongodb:27017/".
- `model_path` : str
    The path where the trained model will be saved, defaulting to "model.h5".

Dependencies
------------
- tensorflow
- pymongo
"""

import os
import tensorflow as tf
from pymongo import MongoClient
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
)
from tensorflow.keras.models import Model
import gridfs
from datetime import datetime
import pika
import json

# Enable mixed precision
tf.keras.mixed_precision.set_global_policy("mixed_float16")

# MongoDB setup
MONGO_URI = "mongodb://mongodb:27017/"
client = MongoClient(MONGO_URI)
db = client.bird_dataset
collection = db.images
fs = gridfs.GridFS(db)
metrics_collection = db.metrics
model_path = "model.keras"

# RabbitMQ setup
rabbitmq_host = os.getenv("RABBITMQ_HOST", "rabbitmq")
rabbitmq_queue = "training_updates"


class MongoDBLogger(Callback):
    def __init__(self):
        super().__init__()
        self.training_id = metrics_collection.insert_one(
            {"timestamp": datetime.now(), "epochs": []}
        ).inserted_id
        self.phase = None
        self.connection = None
        self.channel = None
        self.connect_to_rabbitmq()

    def connect_to_rabbitmq(self):
        if self.connection and self.connection.is_open:
            return
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=rabbitmq_host)
        )
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue=rabbitmq_queue)

    def close_connection(self):
        if self.connection and self.connection.is_open:
            self.connection.close()

    def on_epoch_end(self, epoch, logs=None):
        epoch_metrics = {
            "epoch": epoch,
            "phase": self.phase,
            "loss": logs.get("loss"),
            "accuracy": logs.get("accuracy"),
            "val_loss": logs.get("val_loss"),
            "val_accuracy": logs.get("val_accuracy"),
        }
        metrics_collection.update_one(
            {"_id": self.training_id}, {"$push": {"epochs": epoch_metrics}}
        )

        # Send simple message to RabbitMQ
        message = {
            "training_id": str(self.training_id),
            "epoch": epoch,
            "phase": self.phase,
        }

        self.connect_to_rabbitmq()
        self.channel.basic_publish(
            exchange="", routing_key=rabbitmq_queue, body=json.dumps(message)
        )


def create_model(num_classes: int) -> Model:
    """
    Create and compile a Keras model using InceptionV3 as the base model.

    :param num_classes: The number of output classes.
    :type num_classes: int
    :return: The compiled Keras model.
    :rtype: tensorflow.keras.models.Model

    The model consists of:
    - InceptionV3 base model without the top layer, with weights frozen
    - Global Average Pooling layer
    - Batch Normalization layer
    - Dropout layer with 50% dropout rate
    - Dense output layer with softmax activation and L2 regularization
    """
    base_model = InceptionV3(include_top=False, input_shape=(299, 299, 3))
    base_model.trainable = False

    inputs = tf.keras.layers.Input(shape=(299, 299, 3), name="input-layer")
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D(name="global_average_pooling_layer")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(
        num_classes,
        activation="softmax",
        kernel_regularizer=tf.keras.regularizers.l2(0.01),
        name="output-layer",
    )(x)
    model = Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def parse_tfrecord(tfrecord):
    """
    Parse a single TFRecord example.

    :param tfrecord: The TFRecord example to parse.
    :type tfrecord: tf.train.Example
    :return: A tuple containing the image tensor and the label.
    :rtype: tuple (tensorflow.Tensor, int)

    The function expects TFRecord examples to contain:
    - image: A JPEG-encoded image
    - label: An integer label
    """
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
    }

    example = tf.io.parse_single_example(tfrecord, feature_description)
    image = tf.io.decode_jpeg(example["image"])
    image = tf.image.convert_image_dtype(image, tf.float32)  # Convert to float32
    label = example["label"]
    return image, label


def load_tfrecord_dataset(
    tfrecord_path: str, batch_size: int, shuffle_buffer_size: int, num_classes: int
):
    """
    Load a TFRecord dataset and prepare it for training.

    :param tfrecord_path: The path to the TFRecord file.
    :type tfrecord_path: str
    :param batch_size: The batch size for training.
    :type batch_size: int
    :param shuffle_buffer_size: The buffer size for shuffling the dataset.
    :type shuffle_buffer_size: int
    :param num_classes: The number of output classes.
    :type num_classes: int
    :return: A prepared tf.data.Dataset object.
    :rtype: tensorflow.data.Dataset

    The dataset is:
    - Parsed using `parse_tfrecord`
    - One-hot encoded for the labels
    - Shuffled and batched
    - Prefetched for performance
    """
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(lambda x, y: (x, tf.one_hot(y, num_classes)))
    dataset = (
        dataset.shuffle(shuffle_buffer_size)
        .repeat()
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return dataset


def download_data_and_save_to_tfrecord(split: str, tfrecord_path: str):
    """
    Download data from MongoDB and save it to a TFRecord file.

    :param split: The dataset split to download ('train', 'val', or 'test').
    :type split: str
    :param tfrecord_path: The path where the TFRecord file will be saved.
    :type tfrecord_path: str

    This function:
    - Retrieves image data and labels from MongoDB based on the dataset split
    - Writes the data to a TFRecord file
    """
    data_cursor = list(collection.find({"set_type": split, "image_type": "processed"}))

    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for item in data_cursor:
            image_data = fs.get(item["image_id"]).read()
            label = item["label"]

            feature = {
                "image": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[image_data])
                ),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            }

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())


def train_and_evaluate_model() -> None:
    """
    Train and evaluate the model.

    This function:
    - Downloads and saves the training, validation, and test datasets to TFRecord files
    - Loads the datasets for training and evaluation
    - Creates and trains the model with frozen and then fine-tuned InceptionV3 base
    - Evaluates the model on the test dataset
    - Saves the trained model to a file
    """
    print("Starting model training...")

    batch_size = 16
    shuffle_buffer_size = 2048
    train_tfrecord_path = "train.tfrecord"
    val_tfrecord_path = "val.tfrecord"
    test_tfrecord_path = "test.tfrecord"

    # Download and save datasets to TFRecord
    download_data_and_save_to_tfrecord("train", train_tfrecord_path)
    download_data_and_save_to_tfrecord("val", val_tfrecord_path)
    download_data_and_save_to_tfrecord("test", test_tfrecord_path)

    num_classes = len(
        collection.distinct("label", {"set_type": "train", "image_type": "processed"})
    )

    train_dataset = load_tfrecord_dataset(
        train_tfrecord_path, batch_size, shuffle_buffer_size, num_classes
    )
    val_dataset = load_tfrecord_dataset(
        val_tfrecord_path, batch_size, shuffle_buffer_size, num_classes
    )
    test_dataset = load_tfrecord_dataset(
        test_tfrecord_path, batch_size, shuffle_buffer_size, num_classes
    )

    total_items_train = sum(1 for _ in tf.data.TFRecordDataset(train_tfrecord_path))
    steps_per_epoch_train = total_items_train // batch_size

    total_items_val = sum(1 for _ in tf.data.TFRecordDataset(val_tfrecord_path))
    steps_per_epoch_val = total_items_val // batch_size

    total_items_test = sum(1 for _ in tf.data.TFRecordDataset(test_tfrecord_path))
    steps_per_epoch_test = total_items_test // batch_size

    model = create_model(num_classes)

    # Define callbacks
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=2, min_lr=0.00001
    )

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    mongo_logger = MongoDBLogger()

    print("Training with frozen base model...")
    mongo_logger.phase = "frozen"
    try:
        model.fit(
            train_dataset,
            epochs=2,
            validation_data=val_dataset,
            steps_per_epoch=steps_per_epoch_train,
            validation_steps=steps_per_epoch_val,
            callbacks=[reduce_lr, early_stopping, mongo_logger],
        )
    except Exception as e:
        print("Error during model.fit:", e)

    print("Fine-tuning the model...")
    base_model = model.layers[1]
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    mongo_logger.phase = "unfrozen"
    model.fit(
        train_dataset,
        epochs=2,
        validation_data=val_dataset,
        steps_per_epoch=steps_per_epoch_train,
        validation_steps=steps_per_epoch_val,
        callbacks=[reduce_lr, early_stopping, mongo_logger],
    )

    mongo_logger.close_connection()

    print("Evaluating on test data...")
    test_loss, test_accuracy = model.evaluate(test_dataset, steps=steps_per_epoch_test)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

    # Check if the model file exists and remove it if so
    if os.path.exists(model_path):
        print(f"Removing existing model at {model_path}...")
        os.remove(model_path)

    print("Saving model...")
    model.save(model_path, save_format="keras")


if __name__ == "__main__":
    # Configure TensorFlow for optimal GPU usage
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    train_and_evaluate_model()
