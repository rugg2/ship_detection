import tensorflow as tf
import tf.keras.backend as K

# ------------ define U-NET ------------
def define_up_and_down_stacks(encoder):
    """
    return down_stack, up_stack
    """
    # get multiple outputs from intermediary layers of encoder
    layers_of_encoder_fed_to_decoder = [
        "block2_sepconv2_bn",  # 125 x 125
        "block3_sepconv2_bn",  # 63 x 63
        "block4_sepconv2_bn",  # 32 x 32
        "block13_sepconv2_bn",  # 16 x 16
        "block14_sepconv2_act",  # 8 x 8
    ]

    layers = [
        encoder.get_layer(name).output for name in layers_of_encoder_fed_to_decoder
    ]

    # Xception layers has some weird shapes: returns 125 and 63 for the first 2 selected layers
    paddings_to_correct_size = [
        tf.keras.layers.ZeroPadding2D(((3, 0), (3, 0))),
        tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0))),
        tf.keras.layers.ZeroPadding2D(((0, 0), (0, 0))),
        tf.keras.layers.ZeroPadding2D(((0, 0), (0, 0))),
        tf.keras.layers.ZeroPadding2D(((0, 0), (0, 0))),
    ]

    # Create the feature extraction model
    down_stack = tf.keras.Model(
        inputs=encoder.input,
        outputs=[padding(l) for l, padding in zip(layers, paddings_to_correct_size)],
    )
    down_stack.trainable = False

    # Unlike the UpSampling2D layer, the Conv2DTranspose will learn during training and will attempt to fill in detail as part of the upsampling process.
    # --> here with strides of 2, we'll double the image dimension at each layer
    up_stack = [
        tf.keras.layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding="same"),
        tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same"),
        tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same"),
        tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same"),
    ]

    return down_stack, up_stack


def unet_model(encoder, unet_input_shape=None, output_channels=2):
    inputs = tf.keras.layers.Input(shape=[unet_input_shape, unet_input_shape, 3])
    x = inputs

    # define connections betweem encoder and decoder
    down_stack, up_stack = define_up_and_down_stacks(encoder)

    # Downsampling through the model
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
        output_channels, 3, strides=2, padding="same"
    )  # 64x64 -> 128x128

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


# ----------- metrics and loss -----------
def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2.0 * intersection + smooth) / (union + smooth), axis=0)


def dice_p_bce(in_gt, in_pred):
    return 1e-3 * tf.keras.losses.binary_crossentropy(in_gt, in_pred) - dice_coef(
        in_gt, in_pred
    )


def true_positive_rate(y_true, y_pred):
    return K.sum(K.flatten(y_true) * K.flatten(K.round(y_pred))) / K.sum(y_true)


# --------- script: TODO this bit still to be refactored + define main ---------
def training_main(cur_gen, valid_gen):
    # load encoder
    # step 1: load pretrained encoder
    encoder_and_classifier = tf.keras.models.load_model(
        "../input/vessel-detection-transferlearning-xception/model_xception_gmp_cycling_20200112_7_40.h5"
    )

    encoder = encoder_and_classifier.get_layer("xception")

    # generate unet
    unet = unet_model(encoder, unet_input_shape=None, output_channels=2)

    # visualise unet
    tf.keras.utils.plot_model(unet, show_shapes=True)

    # compile model
    unet.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4, decay=1e-6),
        loss=dice_p_bce,
        metrics=[
            dice_coef,
            "binary_accuracy",
            #                       true_positive_rate
        ],
    )

    # call backs
    # from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
    # weight_path="{}_weights.best.hdf5".format('seg_model')

    # checkpoint = ModelCheckpoint(weight_path, monitor='val_dice_coef', verbose=1,
    #                              save_best_only=True, mode='max', save_weights_only = True)

    # reduceLROnPlat = ReduceLROnPlateau(monitor='val_dice_coef', factor=0.5,
    #                                    patience=3,
    #                                    verbose=1, mode='max', epsilon=0.0001, cooldown=2, min_lr=1e-6)
    # early = EarlyStopping(monitor="val_dice_coef",
    #                       mode="max",
    #                       patience=15) # probably needs to be more patient, but kaggle time is limited
    # callbacks_list = [checkpoint, early, reduceLROnPlat]

    # get a fixed validation batch (not a must, just a choice here)
    # TODO: get cur_gen and valid_gen from preprocessing
    valid_x, valid_y = next(valid_gen)

    # training
    loss_history = [
        unet.fit_generator(
            cur_gen,
            steps_per_epoch=10,
            epochs=5,
            validation_data=(valid_x, valid_y),
            validation_steps=1,
            # callbacks=callbacks_list,
            workers=1,  # the generator is not very thread safe
        )
    ]
