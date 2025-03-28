import os

import tensorflow as tf
from keras import models, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.callbacks import EarlyStopping, ModelCheckpoint

from utils import MODELS_PATH, DATASETS_PATH


if __name__ == '__main__':
    os.makedirs(MODELS_PATH, exist_ok=True)
    
    batch_size = 64

    datagen = ImageDataGenerator()
    train_generator = datagen.flow_from_directory(
        directory=f"{DATASETS_PATH}/tfds/train",
        target_size=(224, 224),
        color_mode="rgb",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
    )

    val_generator = datagen.flow_from_directory(
        directory=f"{DATASETS_PATH}/tfds/validation",
        target_size=(224, 224),
        color_mode="rgb",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
    )

    test_generator = datagen.flow_from_directory(
        directory=f"{DATASETS_PATH}/tfds/test",
        target_size=(224, 224),
        color_mode="rgb",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    conv_base = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    conv_base.trainable = False

    model = models.Sequential()

    model.add(conv_base)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(8, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer="adam",
        metrics=['acc']
    )

    model.summary()

    callbacks = [
        EarlyStopping(patience=3),
        ModelCheckpoint(
            f"{MODELS_PATH}/best_model.keras", 
            save_best_only=True
        )
    ]

    history = model.fit(
        train_generator,
        epochs          = 10,
        steps_per_epoch = 30,
        validation_data = val_generator,
        validation_steps= 10,
        verbose         = 2,
        callbacks       = callbacks,
    )
