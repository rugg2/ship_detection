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
    ]  # corrupted images

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
# def rle_to_pixels(rle_code):
#     """
#     RLE: Run-Length Encoding
#     Decode box position
#     Transforms a RLE code string into a list of pixels of a (768, 768) canvas.

#     Source: https://www.kaggle.com/julian3833/2-understanding-and-plotting-rle-bounding-boxes
#     """
#     rle_code = [int(i) for i in rle_code.split()]
#     pixels = [
#         (pixel_position % 768, pixel_position // 768)
#         for start, length in list(zip(rle_code[0:-1:2], rle_code[1::2]))
#         for pixel_position in range(start, start + length)
#     ]
#     return pixels

# def select_images_with_vessels_and_generate_pixels_mask(metadata_filepath):
#     """
#     This method associates images with pixel masks of where the ship is.

#     Context: this prepares the second part of training:
#     assuming there is a vessel on the image, and predicting where the vessel is.
#     """

#     # load
#     df_csv = pd.read_csv(metadata_filepath)

#     # limit to images with vessels
#     mask_has_vessel = df_csv["EncodedPixels"].notnull()

#     # remove corrupted images. Source: https://www.kaggle.com/iafoss/fine-tuning-resnet34-on-ship-detection
#     exclude_list = [
#         "6384c3e78.jpg",
#         "13703f040.jpg",
#         "14715c06d.jpg",
#         "33e0ff2d5.jpg",
#         "4d4e09f2a.jpg",
#         "877691df8.jpg",
#         "8b909bb20.jpg",
#         "a8d99130e.jpg",
#         "ad55c3143.jpg",
#         "c8260c541.jpg",
#         "d6c7f17c7.jpg",
#         "dc3e7c901.jpg",
#         "e44dffe88.jpg",
#         "ef87bad36.jpg",
#         "f083256d8.jpg",
#     ]

#     mask_not_corrupted = ~(df_csv["ImageId"].isin(exclude_list))

#     # decode masks
#     df_ship_pixel_masks = df_csv.loc[(mask_has_vessel & mask_not_corrupted)].copy()
#     df_ship_pixel_masks["pixel_mask"] = df_ship_pixel_masks["EncodedPixels"].apply(
#         rle_to_pixels
#     )

#     return df_ship_pixel_masks


def rle_decode(mask_rle, shape=(768, 768)):
    """
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
    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros((768, 768), dtype=np.int16)

    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)
    return np.expand_dims(all_masks, -1)


TRAIN_IMAGE_DIR = "../input/airbus-ship-detection/train_v2/"


def make_image_gen(in_df, batch_size=20):
    all_batches = list(in_df.groupby("ImageId"))
    out_rgb = []
    out_mask = []
    while True:
        np.random.shuffle(all_batches)
        for c_img_id, c_masks in all_batches:
            rgb_path = os.path.join(TRAIN_IMAGE_DIR, c_img_id)
            c_img = imread(rgb_path)
            c_mask = masks_as_image(c_masks["EncodedPixels"].values)
            if IMG_SCALING is not None:
                c_img = c_img[:: IMG_SCALING[0], :: IMG_SCALING[1]]
                c_mask = c_mask[:: IMG_SCALING[0], :: IMG_SCALING[1]]
            out_rgb += [c_img]
            out_mask += [c_mask]
            if len(out_rgb) >= batch_size:
                yield np.stack(out_rgb, 0) / 255.0, np.stack(out_mask, 0)
                out_rgb, out_mask = [], []


# use methods used in https://www.kaggle.com/kmader/baseline-u-net-model-part-1

# Augment Data
from keras.preprocessing.image import ImageDataGenerator

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
# brightness can be problematic since it seems to change the labels differently from the images
# if AUGMENT_BRIGHTNESS:
#     dg_args[' brightness_range'] = [0.5, 1.5]
image_gen = ImageDataGenerator(**dg_args)

# if AUGMENT_BRIGHTNESS:
#     dg_args.pop('brightness_range')
label_gen = ImageDataGenerator(**dg_args)

training_image_and_label_gen = MergedGenerators(image_gen, label_gen)

def create_aug_gen(in_gen, seed=None):
    np.random.seed(seed if seed is not None else np.random.choice(range(9999)))
    for in_x, in_y in in_gen:
        seed = np.random.choice(range(9999))
        # keep the seeds syncronized otherwise the augmentation to the images is different from the masks
        g_x = image_gen.flow(
            255 * in_x, batch_size=in_x.shape[0], seed=seed, shuffle=True
        )
        g_y = label_gen.flow(in_y, batch_size=in_x.shape[0], seed=seed, shuffle=True)

        yield next(g_x) / 255.0, next(g_y)


train_gen = make_image_gen(balanced_train_df)
cur_gen = create_aug_gen(train_gen)
t_x, t_y = next(cur_gen)
print("x", t_x.shape, t_x.dtype, t_x.min(), t_x.max())
print("y", t_y.shape, t_y.dtype, t_y.min(), t_y.max())
# only keep first 9 samples to examine in detail
t_x = t_x[:9]
t_y = t_y[:9]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.imshow(montage_rgb(t_x), cmap="gray")
ax1.set_title("images")
ax2.imshow(montage(t_y[:, :, :, 0]), cmap="gray_r")
ax2.set_title("ships")

from keras.utils import Sequence


class MergedGenerators(Sequence):
    def __init__(self, *generators):
        self.generators = generators
        # TODO add a check to verify that all generators have the same length

    def __len__(self):
        return len(self.generators[0])

    def __getitem__(self, index):
        return [generator[index] for generator in self.generators]


# def augmentationForTrainImageAndMask(imgs, masks):
#     data_gen_args = dict(
#         rotation_range=40.0,
#         width_shift_range=0.1,
#         height_shift_range=0.1,
#         zoom_range=0.2,
#         horizontal_flip=True,
#         fill_mode="nearest",
#     )
#     image_datagen = ImageDataGenerator(**data_gen_args)
#     mask_datagen = ImageDataGenerator(**data_gen_args)

#     seed = 1
#     image_datagen.fit(imgs, augment=True, seed=seed)
#     mask_datagen.fit(masks, augment=True, seed=seed)

#     image_generator = image_datagen.flow(
#         imgs, seed=seed, batch_size=batch_size, shuffle=False
#     )

#     mask_generator = mask_datagen.flow(
#         masks, seed=seed, batch_size=batch_size, shuffle=False
#     )

#     # return zip(image_generator, mask_generator)

#     return MergedGenerators(image_generator, mask_generator)

