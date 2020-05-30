# --------------------- not used anywhere yet ---------------------
# https://github.com/keras-team/keras/issues/3059
# https://keras.io/preprocessing/image/

# data_gen_args = dict(
#     featurewise_center=True,
#     featurewise_std_normalization=True,
#     rotation_range=90.0,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     zoom_range=0.2,
# )
# image_datagen = ImageDataGenerator(**data_gen_args)
# mask_datagen = ImageDataGenerator(**data_gen_args)

# # Provide the same seed and keyword arguments to the fit and flow methods
# seed = 1
# image_datagen.fit(images, augment=True, seed=seed)
# mask_datagen.fit(masks, augment=True, seed=seed)

# image_generator = image_datagen.flow_from_directory(
#     "data/images", class_mode=None, seed=seed
# )

# mask_generator = mask_datagen.flow_from_directory(
#     "data/masks", class_mode=None, seed=seed
# )

# # combine generators into one which yields image and masks
# train_generator = zip(image_generator, mask_generator)

# model.fit_generator(train_generator, steps_per_epoch=2000, epochs=50)


data_gen_args = dict(width_shift_range=0.2, height_shift_range=0.2)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)
seed = 1
image_generator = image_datagen.flow_from_directory(
    "raw/train_image",
    target_size=(img_rows, img_cols),
    class_mode=None,
    seed=seed,
    batch_size=batchsize,
    color_mode="grayscale",
)
mask_generator = mask_datagen.flow_from_directory(
    "raw/train_mask",
    target_size=(img_rows, img_cols),
    class_mode=None,
    seed=seed,
    batch_size=batchsize,
    color_mode="grayscale",
)

train_generator = zip(image_generator, mask_generator)

model.fit_generator(
    train_generator, steps_per_epoch=5635 / batchsize, epochs=100, verbose=1
)


# --------------------- other idea ------------------
# https://keras.io/preprocessing/image/


def augmentationForTrainImageAndMask(imgs, masks):
    data_gen_args = dict(
        rotation_range=40.0,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    seed = 1
    image_datagen.fit(imgs, augment=True, seed=seed)
    mask_datagen.fit(masks, augment=True, seed=seed)

    image_generator = image_datagen.flow(
        imgs, seed=seed, batch_size=batch_size, shuffle=False
    )

    mask_generator = mask_datagen.flow(
        masks, seed=seed, batch_size=batch_size, shuffle=False
    )

    return zip(image_generator, mask_generator)


# ------------------- a thread safe / parallelisable solution ------------------
from keras.utils import Sequence


class MergedGenerators(Sequence):
    def __init__(self, *generators):
        self.generators = generators
        # TODO add a check to verify that all generators have the same length

    def __len__(self):
        return len(self.generators[0])

    def __getitem__(self, index):
        return [generator[index] for generator in self.generators]


train_generator = MergedGenerators(image_generator, mask_generator)


# --------------------- using FLOW ----------------------
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

# -------------------- custom Keras Sequence class for masks ----------------------
# https://towardsdatascience.com/keras-data-generators-and-how-to-use-them-b69129ed779c

