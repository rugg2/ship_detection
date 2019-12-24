import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# --------------- show random image ---------------
def show_random_image(
    ship=True, df_csv, input_dir="../../../datasets/satellite_ships"
):
    if ship:
        mask = df_csv["EncodedPixels"].notnull()
    else:
        mask = df_csv["EncodedPixels"].isnull()

    segmentation = df_csv[mask].sample().iloc[0]

    # note: to use plt.imread, need to install not only matplotlib but also "Pillow"
    image = plt.imread(input_dir + "/train_v2/" + segmentation["ImageId"])

    fig = plt.figure(figsize=(16, 8))
    plt.imshow(image)


# --------------- decode box position -----------------
def rle_to_pixels(rle_code):
    """
    RLE: Run-Length Encoding
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


def process_text_df():
    # load
    df_csv = pd.read_csv(
        "../input/airbus-ship-detection/train_ship_segmentations_v2.csv"
    )

    # does image have vessel
    df_csv["has_vessel"] = df_csv["EncodedPixels"].notnull()

    # where is ship on image

    return df_csv


# images with multiple vessels have multiple rows
# most images have no vessels - 77% in fact
# some images have up to 15 vessels
# df_csv.groupby('ImageId')['has_vessel'].sum().describe([0.5, 0.77, 0.78, 0.9, 0.95, 0.98, 0.99])

# ----- example visualisation -----
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

