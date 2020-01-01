from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

def define_model_supersimple_convnet(IMG_HEIGHT, IMG_WIDTH):
    model = keras.Sequential(
        [
            keras.layers.Conv2D(
                16, 3, padding="same", activation="relu", input_shape=(256, 256, 3)
            ),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            keras.layers.MaxPooling2D(),
            keras.layers.Dropout(0.2),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(2, activation="softmax")
            #     keras.layers.Dense(1, activation = 'sigmoid')
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-4
        ),  # update to 3E-4 in the future # this LR is overriden by base cycle LR if CyclicLR callback used
        loss="sparse_categorical_crossentropy",
        #               loss='binary blabla
        metrics=["accuracy"],
    )

    print(model.summary())

    return model

# >>>>>>
# ------------------ notes for model training ------------------
# >>>>>>
def elements_for_model_training(model, train_generator, validation_generator):
    """
    Note to the reader:
    In practice this step often quite a bit of babysitting, so I tend to run these elements in a notebook 

    I have included some simple code snipets here for completeness only, 
    but this is by no means exhaustive or representative of the actual training process often 
    - training is compatible with production code if the model is serialised as a one-off
    - if the model had to be trained in production, I'd recommend documenting the process
    """

    # ------- checkpoint callback -------
    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        # save_weights_only=True,
        save_best_only=True,
        verbose=1,
    )

    # ------- tensorboard callback -------
    import datetime
    log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # ------- learning rate finder -------
    from learning_rate_utils import LRFinder
    lr_finder = LRFinder(start_lr=1e-7, end_lr=1, max_steps=1000)

    # ------- cycling learning rate -------
    from learning_rate_utils import CyclicLR
    # step_size is the number of iteration per half cycle
    # authors suggest setting step_size to 2-8x the number of training iterations per epoch
    cyclic_learning_rate = CyclicLR(base_lr=1E-5, max_lr=1E-2,
                            step_size=5000, mode='triangular2')   

    # ------- actual_training -------
    # model.fit(next(train_data_gen)[0], next(train_data_gen)[1], epochs=20)

    history = model.fit_generator(
            train_generator,
    #         train_example_gen,
    #         40 img/batch * 1000 steps per epoch * 20 epochs = 800k = 200k*4 --> see all data points + their 3 flipped versions once on average 
            steps_per_epoch=1000,
            epochs=35,
            validation_data=validation_generator,
            validation_steps=100,
            # initial_epoch=25,
            callbacks=[
                cp_callback, 
    #             lr_finder,
    #             cyclic_learning_rate,
    #              tensorboard_callback            
            ]

    # ------- save -------
    # TODO: save history variable: often pretty useful retrospectively

    # TODO: save model when results meet expectations

    # ------- plot learning curves -------
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")
    plt.show()
