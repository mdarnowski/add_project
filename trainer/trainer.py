import gc
import io
from typing import Tuple, Iterator
import gridfs
import keras.backend
import numpy as np
import tensorflow as tf
from PIL import Image
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
from tensorflow.keras import mixed_precision
from tensorflow.keras.utils import to_categorical
import itertools

# Enable mixed precision
mixed_precision.set_global_policy("mixed_float16")

# MongoDB setup
MONGO_URI = "mongodb://127.0.0.1:27017/"
client = MongoClient(MONGO_URI)
db = client.bird_dataset
collection = db.images
fs = gridfs.GridFS(db)


def get_num_classes() -> int:
    """Fetch the number of unique classes from the database."""
    num_classes = len(collection.distinct("label", {"image_type": "processed"}))
    print(f"Found {num_classes} unique classes in the dataset.")
    return num_classes


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
        dtype=tf.float32,  # Ensure final layer is float32 for stability
    )(x)
    model = Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def data_generator(
    split: str, batch_size: int, num_classes: int
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    def generator():
        data_cursor = collection.find({"set_type": split, "image_type": "processed"})
        X, y = [], []
        for item in data_cursor:
            # Read the image
            image_data = fs.get(item["image_id"]).read()
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            img_array = np.array(image) / 255.0  # Normalize to [0, 1]

            # Get the label directly
            label = item["label"]

            X.append(img_array)
            y.append(label)
            if len(X) == batch_size:
                X_batch = np.array(X)
                y_batch = to_categorical(np.array(y), num_classes=num_classes)
                yield X_batch, y_batch
                X, y = [], []  # Reset for the next batch
        if len(X) > 0:
            X_batch = np.array(X)
            y_batch = to_categorical(np.array(y), num_classes=num_classes)
            yield X_batch, y_batch

    return itertools.cycle(generator())


def count_samples(split: str) -> int:
    sample_count = collection.count_documents(
        {"set_type": split, "image_type": "processed"}
    )
    print(f"Found {sample_count} samples for split '{split}'.")
    return sample_count


def train_and_evaluate_model() -> None:
    print("Starting model training...")

    batch_size = 16
    num_classes = get_num_classes()
    print(f"Number of classes: {num_classes}")

    train_steps = count_samples("train") // batch_size
    val_steps = count_samples("val") // batch_size

    train_generator = data_generator("train", batch_size, num_classes)
    val_generator = data_generator("val", batch_size, num_classes)

    model = create_model(num_classes)

    # Define callbacks
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=2, min_lr=0.00001
    )
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    print("Training with frozen base model...")
    model.fit(
        train_generator,
        steps_per_epoch=train_steps,
        epochs=2,
        validation_data=val_generator,
        validation_steps=val_steps,
        callbacks=[reduce_lr, early_stopping],
    )

    print("Fine-tuning the model...")
    base_model = model.layers[1]  # Assuming base model is the second layer
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_generator,
        steps_per_epoch=train_steps,
        epochs=20,
        validation_data=val_generator,
        validation_steps=val_steps,
        callbacks=[reduce_lr, early_stopping],
    )
    gc.collect()
    keras.backend.clear_session()
    print("Model training complete. Loading test data...")
    test_generator = data_generator("test", batch_size, num_classes)
    test_steps = count_samples("test") // batch_size
    print("Evaluating on test data...")
    test_loss, test_accuracy = model.evaluate(test_generator, steps=test_steps)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
    print("Saving model...")
    model.save("model.h5")


if __name__ == "__main__":
    train_and_evaluate_model()
