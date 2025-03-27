import os

from keras import models
from keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16

from utils import GIT_ROOT


if __name__ == '__main__':
    if not os.path.exists(f"{GIT_ROOT}/models"):
        os.makedirs(f"{GIT_ROOT}/models")
    
    batch_size = 64
    DATASETS_PATH = f"{GIT_ROOT}/datasets/"

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
        shuffle=True,
    )

    conv_base = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    conv_base.trainable = False

    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(8, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer="adam",
        metrics=['acc']
    )

    model.summary()

    history = model.fit(
        train_generator,
        epochs          = 10,
        steps_per_epoch = 30,
        validation_data = val_generator,
        validation_steps= 10,
        verbose         = 2,
    )

    model_path = f"{GIT_ROOT}/models/my_model.keras"
    print(f"Saving model at {model_path}")
    model.save(model_path)
