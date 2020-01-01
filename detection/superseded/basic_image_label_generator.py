import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# magic numbers
input_dir = "../../../datasets/satellite_ships"
IMG_WIDTH = 768
IMG_HEIGHT = 768
IMG_CHANNELS = 3
TARGET_WIDTH = 256
TARGET_HEIGHT = 256


def get_image(image_name):
    img = plt.imread(input_dir + "/train_v2/" + image_name)  # [:,:,:IMG_CHANNELS]
    #     plt.imshow(img)
    img = tf.image.resize(img, (TARGET_WIDTH, TARGET_HEIGHT), antialias=True)
    img = img / 255
    return img


def create_image_generator(BATCH_SIZE, df_metadata_from_csv):
    #     while True:
    for k, group_df in df_metadata_from_csv.groupby(
        np.arange(df_metadata_from_csv.shape[0]) // BATCH_SIZE
    ):
        imgs = []
        labels = []
        for ImageId in group_df["ImageId"].unique():
            # image
            original_img = get_image(ImageId)
            # label (boat True or False)
            label = (
                group_df.loc[group_df["ImageId"] == ImageId, "EncodedPixels"]
                .notnull()
                .sum()
                > 0
            )

            print("label", label)
            imgs.append(original_img)
            labels.append(label)
        print(k)
        yield imgs, labels


# ----- test script / for notebook -----
# from ./preprocessing import process_text_df
# df_csv = process_text_df(
#     metadata_filepath="../input/airbus-ship-detection/train_ship_segmentations_v2.csv"
# )
generator_example = create_image_generator(BATCH_SIZE=23, df_metadata_from_csv=df_csv)
batch_example = next(generator_example)

i = 0
plt.figure(figsize=(15, 10))
for img, label in zip(*batch_example):
    i += 1
    plt.subplot(2, 5, i)
    plt.title(label)
    plt.imshow(img)
