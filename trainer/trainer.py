import os
import json
import base64
import time
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
)
from tensorflow.keras.models import Model
from loguru import logger
from datetime import datetime
import pika

tf.keras.mixed_precision.set_global_policy("mixed_float16")
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
MODEL_PATH = os.getenv("MODEL_PATH", "model.keras")


class RabbitMQLoggerCallback(Callback):
    def __init__(self):
        super().__init__()
        self.training_id = str(datetime.now())
        self.phase = None
        self.connection = None
        self.channel_uploader = None
        self.channel_presenter = None
        self.queue_uploader = "training_metrics_queue"
        self.queue_presenter = "training_updates"
        self.connect_to_rabbitmq()

    def connect_to_rabbitmq(self):
        if self.connection and self.connection.is_open:
            return
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=RABBITMQ_HOST)
        )
        self.channel_uploader = self.connection.channel()
        self.channel_uploader.queue_declare(queue=self.queue_uploader)
        self.channel_presenter = self.connection.channel()
        self.channel_presenter.queue_declare(queue=self.queue_presenter)

    def close_connection(self):
        if self.connection and self.connection.is_open:
            self.connection.close()

    def on_epoch_end(self, epoch, logs=None):
        epoch_metrics = {
            "training_id": self.training_id,
            "epoch": epoch,
            "phase": self.phase,
            "loss": logs.get("loss"),
            "accuracy": logs.get("accuracy"),
            "val_loss": logs.get("val_loss"),
            "val_accuracy": logs.get("val_accuracy"),
        }
        print("Epoch metrics: sending", epoch_metrics)

        # Send metrics to RabbitMQ
        self.connect_to_rabbitmq()
        self.channel_uploader.basic_publish(
            exchange="",
            routing_key=self.queue_uploader,
            body=json.dumps(epoch_metrics),
        )
        message = {
            "training_id": str(self.training_id),
            "epoch": epoch,
            "phase": self.phase,
        }

        self.channel_presenter.basic_publish(
            exchange="", routing_key=self.queue_presenter, body=json.dumps(message)
        )


class Trainer:
    def __init__(self) -> None:
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
        while True:
            try:
                self.connection = pika.BlockingConnection(
                    pika.ConnectionParameters(self.rabbitmq_host)
                )
                self.channel = self.connection.channel()
                self.channel.queue_declare(queue="image_queue_downloader")
                self.channel.queue_declare(queue="data_request_queue")
                self.channel.queue_declare(queue="num_classes_queue")
                self.channel.queue_declare(queue="training_queue")
                logger.info("Connected to RabbitMQ.")
                # Start consuming num_classes before requesting data
                self.channel.basic_consume(
                    queue="num_classes_queue",
                    on_message_callback=self.receive_num_classes,
                    auto_ack=True,
                )
                self.channel.basic_consume(
                    queue="training_queue",
                    on_message_callback=self.start_training_callback,
                    auto_ack=True,
                )
                break
            except pika.exceptions.AMQPConnectionError:
                logger.warning("RabbitMQ not available, retrying in 5 seconds...")
                time.sleep(5)

    def consume_message(self, ch, method, properties, body) -> None:
        try:
            message = json.loads(body)
            self._process_message(message)
        except Exception as e:
            logger.error(f"Error processing message: {e}")

    def receive_num_classes(self, ch, method, properties, body) -> None:
        try:
            message = json.loads(body)
            self.num_classes = message["num_classes"]
            logger.info(f"Received num_classes: {self.num_classes}")
        except Exception as e:
            logger.error(f"Error receiving num_classes: {e}")

    def send_request(self, split: str) -> None:
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
        feature = {
            "image": tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[image_data])
            ),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        self.tfrecord_writers[split].write(example.SerializeToString())

    def check_all_done(self) -> None:
        if all(self.done_signals.values()) and self.num_classes is not None:
            logger.info("All splits received 'done' signal. Training ready.")
            self.train_and_evaluate_model()

    def start_consuming(self) -> None:
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

    def start_training_callback(self, ch, method, properties, body) -> None:
        logger.info("Received start training signal. Starting training process...")
        self.send_request("train")
        self.send_request("val")
        self.send_request("test")

    def create_model(self) -> Model:
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

        mongo_logger = RabbitMQLoggerCallback()

        print("Training with frozen base model...")
        mongo_logger.phase = "frozen"
        try:
            model.fit(
                train_dataset,
                epochs=3,
                validation_data=val_dataset,
                steps_per_epoch=steps_per_epoch_train,
                validation_steps=steps_per_epoch_val,
                callbacks=[reduce_lr, early_stopping, mongo_logger],
            )
        except Exception as e:
            print("Error during model.fit:", e)

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

        mongo_logger.phase = "unfrozen"
        model.fit(
            train_dataset,
            epochs=3,
            validation_data=val_dataset,
            steps_per_epoch=steps_per_epoch_train,
            validation_steps=steps_per_epoch_val,
            callbacks=[reduce_lr, early_stopping, mongo_logger],
        )

        mongo_logger.close_connection()

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
    # Start consuming messages
    trainer.start_consuming()
