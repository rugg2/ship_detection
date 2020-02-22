import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# --------------- show random image ---------------
def show_random_image(
    df_metadata_from_csv, ship=True, input_dir="../../../datasets/satellite_ships"
):
    if ship:
        mask = df_metadata_from_csv["EncodedPixels"].notnull()
    else:
        mask = df_metadata_from_csv["EncodedPixels"].isnull()

    segmentation = df_metadata_from_csv[mask].sample().iloc[0]

    # note: to use plt.imread, need to install not only matplotlib but also "Pillow"
    print(segmentation["ImageId"])
    image = plt.imread(input_dir + "/train_v2/" + segmentation["ImageId"])

    fig = plt.figure(figsize=(20, 10))
    plt.imshow(image)


def show_model_predictions(validation_generator, model):
    evaluation_batch = next(validation_generator)

    predicted_vessel = model.predict_classes(evaluation_batch)

    print(
        "accuracy on selected batch: ", (evaluation_batch[1] == predicted_vessel).mean()
    )

    # show batch images with their label
    i = 0
    plt.figure(figsize=(15, 40))
    for img, label in zip(*evaluation_batch):
        caption = (
            "prediction:"
            + str(bool(predicted_vessel[i]))
            + ", actual:"
            + str(bool(label))
        )
        i += 1
        plt.subplot(7, 3, i)
        plt.title(caption)
        plt.imshow(img)

        if i > 20:
            break


def visualise_image_and_mask(df_ship_pixel_masks, img_nbr):
    import matplotlib.pyplot as plt

    image = plt.imread(
        "../input/airbus-ship-detection/train_v2/"
        + df_ship_pixel_masks["ImageId"].iloc[img_nbr]
    )

    fig = plt.figure(figsize=(16, 16))
    fig.add_subplot(2, 2, 1)

    plt.imshow(image)
    fig.add_subplot(2, 2, 2)

    decoded_mask = df_ship_pixel_masks["pixel_mask"].iloc[img_nbr]

    canvas = np.zeros(image.shape[0:2])
    canvas[tuple(zip(*decoded_mask))] = 1

    plt.imshow(canvas)


# ----------- exploratory data analysis -----------
# images with multiple vessels have multiple rows
# most images have no vessels - 77% in fact
# some images have up to 15 vessels
# df_csv.groupby('ImageId')['has_vessel'].sum().describe([0.5, 0.77, 0.78, 0.9, 0.95, 0.98, 0.99])

# ----------- example visualisation -----------
# canvas = np.zeros((768, 768))
# pixels = rle_to_pixels(np.random.choice(df_csv.loc[mask_hasvessel, 'EncodedPixels']))
# canvas[tuple(zip(*pixels))] = 1
# plt.imshow(canvas);

# df = df_csv.iloc[3:5].groupby("ImageId")[['EncodedPixels']].agg(lambda rle_codes: ' '.join(rle_codes)).reset_index()

# import PIL
# load_img = lambda filename: np.array(PIL.Image.open(f"../input/train_v2/{filename}"))
# def apply_mask(image, mask):
#     for x, y in mask:
#         image[x, y, [0, 1]] = 255
#     return image
# img = load_img(df.loc[0, 'ImageId'])
# mask_pixels = rle_to_pixels(df.loc[0, 'EncodedPixels'])
# img = apply_mask(img, mask_pixels)
# plt.imshow(img);

# # path_image = '/kaggle/input/airbus-ship-detection/train_v2/4bb14e335.jpg'
# path_image = '../input/airbus-ship-detection/train_v2/4bb14e335.jpg'
# # !ls path_image
# # def show_pic(path)
# plt.figure(figsize=(14,8))
# image = plt.imread(path_image)
# plt.imshow(image)

