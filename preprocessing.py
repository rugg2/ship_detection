import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from skimage.util import montage
from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator


def process_text_df(metadata_filepath):
    """
    Extract labels from metadata csv file.

    Output:
    - dataframe of image file names and boolean of whether there is >= 1 ship or not
    (deduplicated on image file names)
    - dataframe of image file names filtered to those with >= 1 ship 
    (multiple references to the same image if multiple ships)
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
    ]  # corrupted images

    mask_not_corrupted = ~(df_csv["ImageId"].isin(exclude_list))

    df_ship_noship = df_csv.loc[
        mask_not_corrupted, ["has_vessel", "has_vessel_str", "ImageId"]
    ].drop_duplicates()

    df_with_ship = df_csv.loc[mask_not_corrupted & df_csv["has_vessel"]]

    return df_ship_noship, df_with_ship


#  ------------------------ SHIP DETECTION ------------------------
# ---------- image preprocessing for the ship detection task ----------


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
    --> only for vessel detection, not directly usable for localization
    TODO: update method name to reflect that
    """
    df_metadata, _ = process_text_df(
        metadata_filepath=input_dir + "/train_ship_segmentations_v2.csv"
    )

    train_df, dev_df = train_test_split(df_metadata, test_size=0.1, random_state=42)

    train_generator, validation_generator = image_batch_generators(
        train_df, dev_df, target_size=target_size, input_dir=input_dir
    )

    return train_generator, validation_generator


#  ------------------------ SHIP SEGMENTATION ------------------------
# ---------------- preprocess both images and masks -----------------
#  source for decoding and generators: https://www.kaggle.com/kmader/baseline-u-net-model-part-1
# TODO: parameters to pass as argument
TRAIN_IMAGE_DIR = "../input/airbus-ship-detection/train_v2/"

# downsampling in preprocessing, as smaller images train faster and consume less memory
# CAUTION: different definitions of scaling
# IMG_SCALING = (0.5, 0.5)
IMG_SCALING = (2, 2)


def rle_decode(mask_rle, shape=(768, 768)):
    """
    Masks of training set are encoded in a format called RLE (Run Length Encoding)
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    """
    if pd.isnull(mask_rle):
        img = no_mask
        return img.reshape(shape).T
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


def masks_as_image(in_mask_list):
    """
    Take the individual ship masks and create a single mask array for all ships
    """
    all_masks = np.zeros((768, 768), dtype=np.float32)

    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)
    return np.expand_dims(all_masks, -1)


def make_image_gen(
    in_df,
    batch_size=20,
    TRAIN_IMAGE_DIR="../input/airbus-ship-detection/train_v2/",
    IMG_SCALING=(2, 2),
):
    """
    Generators loading both images and masks, as well as performing rescaling
    """
    all_batches = list(in_df.groupby("ImageId"))
    out_rgb = []
    out_mask = []
    while True:
        np.random.shuffle(all_batches)
        for c_img_id, c_rows_with_vessel_masks in all_batches:
            rgb_path = os.path.join(TRAIN_IMAGE_DIR, c_img_id)
            c_img = plt.imread(rgb_path)
            c_mask = masks_as_image(c_rows_with_vessel_masks["EncodedPixels"].values)
            if IMG_SCALING is not None:
                c_img = c_img[:: IMG_SCALING[0], :: IMG_SCALING[1]]
                c_mask = c_mask[:: IMG_SCALING[0], :: IMG_SCALING[1]]
            out_rgb += [c_img]
            out_mask += [c_mask]
            if len(out_rgb) >= batch_size:
                yield np.stack(out_rgb, 0) / 255.0, np.stack(out_mask, 0)
                out_rgb, out_mask = [], []


# AUGMENT DATA: apply a range of distortions
dg_args = dict(
    featurewise_center=False,
    samplewise_center=False,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.01,
    zoom_range=[0.9, 1.25],
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="reflect",
    data_format="channels_last",
)

image_gen = ImageDataGenerator(**dg_args)
label_gen = ImageDataGenerator(**dg_args)


def create_aug_gen(in_gen, seed=None):
    """
    Data augmentation on image and mask/label, from image and mask generators

    Caution: the synchronisation of seeds for image and mask is fragile,
    and does not seem very thread safe, so use only 1 worker.

    TODO: for multithreading, look at keras.utils.Sequence, and the class MergedGenerators
    """
    np.random.seed(seed if seed is not None else np.random.choice(range(9999)))
    for in_x, in_y in in_gen:
        seed = np.random.choice(range(9999))
        # keep the seeds syncronized otherwise the augmentation to the images is different from the masks
        g_x = image_gen.flow(
            255 * in_x, batch_size=in_x.shape[0], seed=seed, shuffle=True
        )
        g_y = label_gen.flow(in_y, batch_size=in_x.shape[0], seed=seed, shuffle=True)

        yield next(g_x) / 255.0, next(g_y)


def split_on_unique_id(df, id_col, test_size=0.1, random_state=42):
    """
    Split dataset into train and dev set, being careful not to split masks relative to the same image
    """
    train_ids, dev_ids = train_test_split(
        df[id_col].drop_duplicates().values,
        test_size=test_size,
        random_state=random_state,
    )

    train_df, test_df = [
        df.loc[df[id_col].isin(subset_ids)] for subset_ids in [train_ids, dev_ids]
    ]
    return train_df, test_df


def preprocessing_segmentation_main(
    input_dir="../../datasets/satellite_ships",
    TRAIN_IMAGE_DIR="../input/airbus-ship-detection/train_v2/",
):
    # to be parametrised
    # TRAIN_IMAGE_DIR
    # VALID_IMG_COUNT

    #  load metadata from csv
    _, df_with_ship = process_text_df(
        metadata_filepath=input_dir + "/train_ship_segmentations_v2.csv"
    )

    # split dataset into train and dev set, being careful not to split masks relative to the same image
    df_images_with_ship_train, df_images_with_ship_dev = split_on_unique_id(
        df=df_with_ship, id_col="ImageId", test_size=0.1, random_state=42
    )

    # generator fetching raw images and masks
    train_gen = make_image_gen(
        in_df=df_images_with_ship_train, TRAIN_IMAGE_DIR=TRAIN_IMAGE_DIR
    )

    # generator augmenting / distorting both images and masks
    cur_gen = create_aug_gen(train_gen)

    # a fixed dev / validation batch
    valid_gen = make_image_gen(
        in_df=df_images_with_ship_dev, TRAIN_IMAGE_DIR=TRAIN_IMAGE_DIR
    )
    # valid_x, valid_y = next(valid_gen)

    # montage_rgb = lambda x: np.stack(
    #     [montage(x[:, :, :, i]) for i in range(x.shape[3])], -1
    # )

    # # plots
    # t_x, t_y = next(cur_gen)
    # print('x', t_x.shape, t_x.dtype, t_x.min(), t_x.max())
    # print('y', t_y.shape, t_y.dtype, t_y.min(), t_y.max())
    # # only keep first 9 samples to examine in detail
    # t_x = t_x[:2]
    # t_y = t_y[:2]
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (30, 15))
    # ax1.imshow(montage_rgb(t_x), cmap='gray')
    # ax1.set_title('images')
    # ax2.imshow(montage(t_y[:, :, :, 0]), cmap='gray_r')
    # ax2.set_title('ships')

    return cur_gen, valid_gen
