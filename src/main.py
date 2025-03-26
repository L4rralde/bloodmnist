from keras import models
from keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16


from utils import GIT_ROOT


if __name__ == '__main__':
    DATASETS_PATH = f"{GIT_ROOT}/datasets/"

    datagen = ImageDataGenerator()
    train_generator = datagen.flow_from_directory(
        directory=f"{DATASETS_PATH}/tfds/train",
        target_size=(224, 224),
        color_mode="rgb",
        batch_size=32,
        class_mode="categorical",
        shuffle=True,
    )

    val_generator = datagen.flow_from_directory(
        directory=f"{DATASETS_PATH}/tfds/validation",
        target_size=(224, 224),
        color_mode="rgb",
        batch_size=32,
        class_mode="categorical",
        shuffle=True,
    )

    test_generator = datagen.flow_from_directory(
        directory=f"{DATASETS_PATH}/tfds/test",
        target_size=(224, 224),
        color_mode="rgb",
        batch_size=32,
        class_mode="categorical",
        shuffle=True,
    )

    conv_base = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )

    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(8, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer="adam",
        metrics=['acc']
    )

    history = model.fit(
        train_generator,
        steps_per_epoch = 100, 
        epochs          = 10,
        validation_data = val_generator,
        validation_steps= 50,
        verbose         = 2,
    )
