import numpy as np
import pandas as pd
import os
import pathlib
import shutil
import tensorflow as tf
import matplotlib.pyplot as plt

AUTOTUNE = tf.data.experimental.AUTOTUNE

# def small_training_set():
#     # this method is not used at the moment

#     mask_has_vessel = df_csv["has_vessel"]
#     df_csv_small_training = pd.concat(
#         [
#             df_csv.loc[mask_has_vessel].iloc[0:50],
#             df_csv.loc[~mask_has_vessel].iloc[0:50],
#         ]
#     )


def organise_images_on_disk(df_csv):
    df_labels = df_csv.groupby("ImageId")["has_vessel"].max()
    df_labels = df_labels.astype(str)

    # data_dir = pathlib.Path('../input/airbus-ship-detection/train_v2
    data_dir_raw = pathlib.Path("../input/airbus-ship-detection/train_v2")

    # execute shell (in notebook works, one per cell - how to do this in python? need to use other methods)
    # mkdir training_small
    # mkdir training_small/ship/
    # mkdir training_small/no_ship/
    # mkdir test_small
    # mkdir test_small/ship/
    # mkdir test_small/no_ship/

    # move those parameters as arguments to the method
    training_size = 450
    test_size = 50

    counter_ship = 0
    counter_no_ship = 0
    for item in data_dir_raw.glob("*.jpg"):
        item_str = str(item)
        if df_labels.loc[item.name] == "True":
            if counter_ship < training_size:
                shutil.copy(item_str, "training_small/ship/")
                counter_ship += 1
            elif counter_ship < training_size + test_size:
                shutil.copy(item_str, "test_small/ship/")
                counter_ship += 1
        else:
            if counter_no_ship < training_size:
                shutil.copy(item_str, "training_small/no_ship/")
                counter_no_ship += 1
            elif counter_no_ship < training_size + test_size:
                shutil.copy(item_str, "test_small/no_ship/")
                counter_no_ship += 1

    # ------ QA ------
    # ls training_small/no_ship/ -1 | wc  -l
    # ls test_small/ship/ -1 | wc  -l
    # ls training_small/ship

    # image_count = len(list(data_dir.glob('*/*.jpg')))
    # image_count


def prepare_tensorflow_from_folder():
    # data_dir = pathlib.Path('../input/airbus-ship-detection/train_v2
    data_dir = pathlib.Path("training_small/")

    # make these parameters of the method
    BATCH_SIZE = 20
    IMG_HEIGHT = 768
    IMG_WIDTH = 768
    # STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)
    epochs = 10

    CLASS_NAMES = np.array([item.name for item in data_dir.glob("*")])

    train = os.listdir("training_small/no_ship")
    print(len(train))

    # The 1./255 is to convert from uint8 to float32 in range [0,1].
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
    validation_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255
    )  # Generator for our validation data

    train_data_gen = image_generator.flow_from_directory(
        directory=str(data_dir),  # training_dir
        batch_size=BATCH_SIZE,
        shuffle=True,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        class_mode="sparse"
        #                                                      classes = list(CLASS_NAMES),
        #                                                     color_mode='grayscale',
        #                                                     data_format='channels_last'
    )

    val_data_gen = validation_image_generator.flow_from_directory(
        directory="test_small",
        batch_size=BATCH_SIZE,
        shuffle=True,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        class_mode="sparse",
    )

    # image_batch, label_batch = next(train_data_gen)
    # print(image_batch.shape)

    # image_batch, label_batch = next(val_data_gen)
    # print(image_batch.shape)

    # print(label_batch.shape)

    return train_data_gen, val_data_gen, epochs, CLASS_NAMES


def show_batch(image_batch, label_batch, CLASS_NAMES):
    plt.figure(figsize=(10, 10))
    for n in range(25):
        # ax = plt.subplot(5, 5, n + 1)
        plt.subplot(5, 5, n + 1)
        plt.imshow(image_batch[n])
        plt.title(CLASS_NAMES[label_batch[n] == 1][0])
        plt.axis("off")

