import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import tensorflow as tf


def process_text_df(metadata_filepath):
    """
    Extract labels from metadata csv file 
    """
    # load
    df_csv = pd.read_csv(metadata_filepath)

    # does image have vessel
    df_csv["has_vessel"] = df_csv["EncodedPixels"].notnull()
    df_csv["has_vessel_str"] = df_csv["has_vessel"].astype(
        str
    )  # for tensorflow flow_from_dataframe generator

    # remove corrupted images. Source: https://www.kaggle.com/iafoss/fine-tuning-resnet34-on-ship-detection
    exclude_list = [
        "6384c3e78.jpg",
        "13703f040.jpg",
        "14715c06d.jpg",
        "33e0ff2d5.jpg",
        "4d4e09f2a.jpg",
        "877691df8.jpg",
        "8b909bb20.jpg",
        "a8d99130e.jpg",
        "ad55c3143.jpg",
        "c8260c541.jpg",
        "d6c7f17c7.jpg",
        "dc3e7c901.jpg",
        "e44dffe88.jpg",
        "ef87bad36.jpg",
        "f083256d8.jpg",
    ]  # corrupted image

    mask_not_corrupted = ~(df_csv["ImageId"].isin(exclude_list))

    df_ship_noship = df_csv.loc[
        mask_not_corrupted, ["has_vessel", "has_vessel_str", "ImageId"]
    ].drop_duplicates()

    return df_ship_noship


def image_batch_generators(
    train_df, dev_df, target_size=(256, 256), input_dir="../../datasets/satellite_ships"
):
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255,
        #         shear_range=0.2,
        #         zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
    )

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=input_dir + "/train_v2/",
        x_col="ImageId",
        y_col="has_vessel_str",
        target_size=target_size,
        batch_size=40,
        class_mode="binary",
    )

    validation_generator = test_datagen.flow_from_dataframe(
        dataframe=dev_df,
        directory=input_dir + "/train_v2/",
        x_col="ImageId",
        y_col="has_vessel_str",
        target_size=target_size,
        batch_size=40,
        class_mode="binary",
    )

    return train_generator, validation_generator


def preprocessing_main(
    target_size=(256, 256), input_dir="../../datasets/satellite_ships"
):
    """
    Call the other subroutines in this file.
    --> likely targetted to vessel detection, not directly usable for localization (tbc)
    """
    df_metadata = process_text_df(
        metadata_filepath=input_dir + "/train_ship_segmentations_v2.csv"
    )

    train_df, dev_df = train_test_split(df_metadata, test_size=0.1, random_state=42)

    train_generator, validation_generator = image_batch_generators(
        train_df, dev_df, target_size=target_size, input_dir=input_dir
    )

    return train_generator, validation_generator


# where is ship on image
# TODO: add when working on the localisation part of the project
def rle_to_pixels(rle_code):
    """
    RLE: Run-Length Encoding
    Decode box position
    Transforms a RLE code string into a list of pixels of a (768, 768) canvas.
    
    Source: https://www.kaggle.com/julian3833/2-understanding-and-plotting-rle-bounding-boxes
    """
    rle_code = [int(i) for i in rle_code.split()]
    pixels = [
        (pixel_position % 768, pixel_position // 768)
        for start, length in list(zip(rle_code[0:-1:2], rle_code[1::2]))
        for pixel_position in range(start, start + length)
    ]
    return pixels
