"""
Trainer Module
==============

This module provides functions for training a deep learning model using TensorFlow and Keras.
It handles data retrieval via RabbitMQ, preparation of datasets in TFRecord format, model creation,
training, and evaluation.

Classes
-------
- Trainer

Dependencies
------------
- tensorflow
- pika
- json
- base64
- os

Environment Variables
---------------------
- RABBITMQ_HOST : str
    The RabbitMQ host address (default: "localhost")
- MODEL_PATH : str
    The path where the trained model will be saved, defaulting to "model.h5".

Functions
---------
- `create_model(num_classes: int) -> tf.keras.models.Model`
    Creates and compiles a Keras model using InceptionV3 as the base model.
- `parse_tfrecord(tfrecord)`
    Parses a single TFRecord example into an image tensor and label.
- `load_tfrecord_dataset(tfrecord_path: str, batch_size: int, shuffle_buffer_size: int, num_classes: int)`
    Loads a TFRecord dataset and prepares it for training.
- `train_and_evaluate_model() -> None`
    Trains and evaluates the model using the downloaded datasets.
"""

import os
import json
import base64
import time

import tensorflow as tf
import pika
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
)
from tensorflow.keras.models import Model
from loguru import logger

# Enable mixed precision
tf.keras.mixed_precision.set_global_policy("mixed_float16")

# Environment Variables
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
MODEL_PATH = os.getenv("MODEL_PATH", "model.keras")


class Trainer:
    """
    Trainer Class
    =============

    This class provides functionality to request image data from RabbitMQ,
    receive and process image data, save to TFRecord, prepare datasets,
    and train and evaluate a deep learning model.

    Methods
    -------
    __init__()
        Initializes the Trainer instance and sets up RabbitMQ connection.
    connect_to_rabbitmq()
        Establishes a connection to RabbitMQ and declares the required queues.
    consume_message(ch, method, properties, body)
        Callback function for consuming image data messages.
    receive_num_classes(ch, method, properties, body)
        Callback function for receiving num_classes.
    send_request(split: str)
        Sends a request message to RabbitMQ for the specified dataset split.
    save_to_tfrecord(split: str, tfrecord_path: str, image_data: bytes, label: int)
        Saves received images incrementally to a TFRecord file.
    start_consuming()
        Starts consuming messages from RabbitMQ.
    train_and_evaluate_model()
        Trains and evaluates the model using the received datasets.
    check_all_done()
        Checks if all splits have received 'done' signals and starts training.
    """

    def __init__(self) -> None:
        """
        Initializes the Trainer instance.

        Sets up RabbitMQ connection and declares queues.

        Environment Variables
        ---------------------
        - RABBITMQ_HOST : str, optional
            The RabbitMQ host address (default: "localhost")
        """
        self.rabbitmq_host = RABBITMQ_HOST
        self.connection = None
        self.channel = None
        self.tfrecord_writers = {
            "train": tf.io.TFRecordWriter("train.tfrecord"),
            "val": tf.io.TFRecordWriter("val.tfrecord"),
            "test": tf.io.TFRecordWriter("test.tfrecord"),
        }
        self.done_signals = {"train": False, "val": False, "test": False}
        self.num_classes = None
        self.connect_to_rabbitmq()

    def connect_to_rabbitmq(self) -> None:
        """
        Establishes a connection to RabbitMQ and declares the required queues.

        Retries the connection every 5 seconds if RabbitMQ is not available.
        """
        while True:
            try:
                self.connection = pika.BlockingConnection(
                    pika.ConnectionParameters(self.rabbitmq_host)
                )
                self.channel = self.connection.channel()
                self.channel.queue_declare(queue="image_queue_downloader")
                self.channel.queue_declare(queue="data_request_queue")
                self.channel.queue_declare(
                    queue="num_classes_queue"
                )  # Added for num_classes
                logger.info("Connected to RabbitMQ.")
                # Start consuming num_classes before requesting data
                self.channel.basic_consume(
                    queue="num_classes_queue",
                    on_message_callback=self.receive_num_classes,
                    auto_ack=True,
                )
                break
            except pika.exceptions.AMQPConnectionError:
                logger.warning("RabbitMQ not available, retrying in 5 seconds...")
                time.sleep(5)

    def consume_message(self, ch, method, properties, body) -> None:
        """
        Callback function for consuming image data messages.

        Parameters
        ----------
        ch : channel
            The channel object.
        method : method
            The method object.
        properties : properties
            The properties object.
        body : bytes
            The message body.
        """
        try:
            message = json.loads(body)
            self._process_message(message)
        except Exception as e:
            logger.error(f"Error processing message: {e}")

    def receive_num_classes(self, ch, method, properties, body) -> None:
        """
        Callback function for receiving num_classes.

        Parameters
        ----------
        ch : channel
            The channel object.
        method : method
            The method object.
        properties : properties
            The properties object.
        body : bytes
            The message body.
        """
        try:
            message = json.loads(body)
            self.num_classes = message["num_classes"]
            logger.info(f"Received num_classes: {self.num_classes}")
        except Exception as e:
            logger.error(f"Error receiving num_classes: {e}")

    def send_request(self, split: str) -> None:
        """
        Sends a request message to RabbitMQ for the specified dataset split.

        Parameters
        ----------
        split : str
            The dataset split to request (e.g., 'train', 'val', 'test').
        """
        try:
            self.channel.basic_publish(
                exchange="",
                routing_key="data_request_queue",
                body=json.dumps({"split": split}),
            )
            logger.info(f"Requested data for {split} split")
        except Exception as e:
            logger.error(f"Error sending request: {e}")

    def _process_message(self, message: dict) -> None:
        """
        Processes the message and saves the image data to TFRecord.

        Parameters
        ----------
        message : dict
            The message containing encoded image data and metadata.
        """
        split = message.get("split")
        if split is None:
            logger.error("Message missing 'split' key.")
            return

        if message.get("status") == "done":
            logger.info(f"Received 'done' signal for {split} split")
            self.tfrecord_writers[split].close()
            self.done_signals[split] = True
            self.check_all_done()
            return

        image_data = base64.b64decode(message["image_data"])
        label = message["label"]

        self.save_to_tfrecord(split, f"{split}.tfrecord", image_data, label)

    def save_to_tfrecord(
        self, split: str, tfrecord_path: str, image_data: bytes, label: int
    ) -> None:
        """
        Saves received images incrementally to a TFRecord file.

        Parameters
        ----------
        split : str
            The dataset split (e.g., 'train', 'val', 'test').
        tfrecord_path : str
            The path where the TFRecord file will be saved.
        image_data : bytes
            The image data in bytes.
        label : int
            The label of the image.
        """
        feature = {
            "image": tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[image_data])
            ),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        self.tfrecord_writers[split].write(example.SerializeToString())

    def check_all_done(self) -> None:
        """
        Checks if all splits have received 'done' signals and starts training.
        """
        if all(self.done_signals.values()) and self.num_classes is not None:
            logger.info("All splits received 'done' signal. Starting training...")
            self.train_and_evaluate_model()

    def start_consuming(self) -> None:
        """
        Starts consuming messages from RabbitMQ.

        Consumes messages from the 'image_queue_downloader' queue and processes them.
        """
        while True:
            try:
                self.channel.basic_consume(
                    queue="image_queue_downloader",
                    on_message_callback=self.consume_message,
                    auto_ack=True,
                )
                logger.info("Trainer is listening for messages...")
                self.channel.start_consuming()
            except (
                pika.exceptions.ConnectionClosed,
                pika.exceptions.AMQPConnectionError,
            ):
                logger.warning("Connection lost, reconnecting...")
                self.connect_to_rabbitmq()

    def create_model(self) -> Model:
        """
        Create and compile a Keras model using InceptionV3 as the base model.

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
            self.num_classes,
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

    def train_and_evaluate_model(self) -> None:
        """
        Train and evaluate the model using the received datasets.

        This function:
        - Loads the training, validation, and test datasets from TFRecord files
        - Creates and trains the model with frozen and then fine-tuned InceptionV3 base
        - Evaluates the model on the test dataset
        - Saves the trained model to a file
        """
        batch_size = 16
        shuffle_buffer_size = 2048
        train_tfrecord_path = "train.tfrecord"
        val_tfrecord_path = "val.tfrecord"
        test_tfrecord_path = "test.tfrecord"

        train_dataset = load_tfrecord_dataset(
            train_tfrecord_path, batch_size, shuffle_buffer_size, self.num_classes
        )
        val_dataset = load_tfrecord_dataset(
            val_tfrecord_path, batch_size, shuffle_buffer_size, self.num_classes
        )
        test_dataset = load_tfrecord_dataset(
            test_tfrecord_path, batch_size, shuffle_buffer_size, self.num_classes
        )

        total_items_train = sum(1 for _ in tf.data.TFRecordDataset(train_tfrecord_path))
        steps_per_epoch_train = total_items_train // batch_size

        total_items_val = sum(1 for _ in tf.data.TFRecordDataset(val_tfrecord_path))
        steps_per_epoch_val = total_items_val // batch_size

        total_items_test = sum(1 for _ in tf.data.TFRecordDataset(test_tfrecord_path))
        steps_per_epoch_test = total_items_test // batch_size

        model = self.create_model()

        # Define callbacks
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=2, min_lr=0.00001
        )

        early_stopping = EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )

        logger.info("Training with frozen base model...")

        try:
            model.fit(
                train_dataset,
                epochs=20,
                validation_data=val_dataset,
                steps_per_epoch=steps_per_epoch_train,
                validation_steps=steps_per_epoch_val,
                callbacks=[reduce_lr, early_stopping],
            )
        except Exception as e:
            logger.error(f"Error during model.fit: {e}")

        logger.info("Fine-tuning the model...")
        base_model = model.layers[1]
        base_model.trainable = True
        for layer in base_model.layers[:-20]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        model.fit(
            train_dataset,
            epochs=20,
            validation_data=val_dataset,
            steps_per_epoch=steps_per_epoch_train,
            validation_steps=steps_per_epoch_val,
            callbacks=[reduce_lr, early_stopping],
        )

        logger.info("Evaluating on test data...")
        test_loss, test_accuracy = model.evaluate(
            test_dataset, steps=steps_per_epoch_test
        )
        logger.info(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

        # Check if the model file exists and remove it if so
        if os.path.exists(MODEL_PATH):
            logger.info(f"Removing existing model at {MODEL_PATH}...")
            os.remove(MODEL_PATH)

        logger.info("Saving model...")
        model.save(MODEL_PATH)


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


if __name__ == "__main__":
    # Configure TensorFlow for optimal GPU usage
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            logger.error(e)

    trainer = Trainer()
    # Send requests for different splits
    trainer.send_request("train")
    trainer.send_request("val")
    trainer.send_request("test")
    # Start consuming messages
    trainer.start_consuming()
