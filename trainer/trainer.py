import os
import tensorflow as tf
from pymongo import MongoClient
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
)
from tensorflow.keras.models import Model
import gridfs

# Enable mixed precision
tf.keras.mixed_precision.set_global_policy("mixed_float16")

# MongoDB setup
MONGO_URI = "mongodb://mongodb:27017/"
client = MongoClient(MONGO_URI)
db = client.bird_dataset
collection = db.images
fs = gridfs.GridFS(db)
model_path = "model.h5"


def create_model(num_classes: int) -> Model:
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


def download_data_and_save_to_tfrecord(split: str, tfrecord_path: str):
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

    # Debug: Inspect the datasets
    try:
        sample = tf.data.experimental.get_single_element(train_dataset)
        print("Sample from train_dataset:", sample)
    except Exception as e:
        print("Error fetching sample from train_dataset:", e)

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

    print("Training with frozen base model...")

    try:
        model.fit(
            train_dataset,
            epochs=2,
            validation_data=val_dataset,
            steps_per_epoch=steps_per_epoch_train,
            validation_steps=steps_per_epoch_val,
            callbacks=[reduce_lr, early_stopping],
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

    model.fit(
        train_dataset,
        epochs=20,
        validation_data=val_dataset,
        steps_per_epoch=steps_per_epoch_train,
        validation_steps=steps_per_epoch_val,
        callbacks=[reduce_lr, early_stopping],
    )

    print("Evaluating on test data...")
    test_loss, test_accuracy = model.evaluate(test_dataset, steps=steps_per_epoch_test)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

    # Check if the model file exists and remove it if so
    if os.path.exists(model_path):
        print(f"Removing existing model at {model_path}...")
        os.remove(model_path)

    print("Saving model...")
    model.save(model_path)


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
