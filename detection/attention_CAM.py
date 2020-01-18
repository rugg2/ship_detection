# Network attention: Class Activation Mapping (CAM)
# inspired from paper https://arxiv.org/pdf/1512.04150.pdf

import tensorflow as tf


def extract_relevant_layers(model, image_batch, global_pooling_layer_nbr=1):
    # --------- CAM LOGIC ---------
    # >>>> define a new variable for the layer just before the Global Pooling layer (GAP/GMP)
    # in our example that's the input to layer 1, or the output of layer 0
    model_pre_gp_layer = tf.keras.models.Model(
        inputs=model.input, outputs=model.layers[global_pooling_layer_nbr].input
    )

    # get an example data, and make a prediction on it
    pre_gp_activation_example = model_pre_gp_layer.predict(batch_test[0])

    # classification weights, last layer (here we are working on binary classification - slight variation for multiclass)
    classification_weights = model.layers[-1].weights[0].numpy()

    return pre_gp_activation_example, classification_weights


def class_activation_mapping(pre_gp_activation_oneimage, classification_weights):
    """
    Calculate weighted sum showing class activation mapping

    Input
    - pre_gmp_activation_oneimage: pre global pooling activations, shaped (height, width, channels)
    - classification_weights: weights for each channel, shaped (channels)
    """

    # this dot product relies on broadcasting on the spatial dimensions
    dot_prod = pre_gp_activation_oneimage.dot(classification_weights)
    return dot_prod


# ------------------------- EXAMPLE USAGE -------------------------
"""
Simplified architecture in the example:
- Image inputs: (batch size=40, height=299, width=299, channels=3)
- Xception output: (batch size=40, height at this layer=10, width at this layer=10, channels at this layer=2048)
- Global Max Pooling output: (batch size=40, 1, 1, channels at this layer=2048)
- Classifier (dense) output: (batch size=40, classification dim=1)

The quantity we show on the heatmap is 
    the Xception output (pre_gmp_activation)
        summed over the channel dimension 
            with the Classifier's weights for each channel (classification_weights).
"""

# --------- INPUT DATA ---------
# get model (here we had pretrained it)
# model_loaded = tf.keras.models.load_model(
#     "../input/vessel-detection-transferlearning-xception/model_xception_gmp_cycling_20200112_7_40.h5"
# )

# get a batch of pictures
# here assuming the validation_generator is running, or through another mean
# batch_test = next(validation_generator)

# --------- PROCESS ---------
# prediction on relevant layers for that batch
pre_gp_activation_batch, classification_weights = extract_relevant_layers(
    model=model_loaded, image_batch=batch_test
)

# select an image and work with this
selected_image_index = 5

dot_prod_oneimage = class_activation_mapping(
    pre_gp_activation_oneimage=pre_gp_activation_batch[selected_image_index],
    classification_weights=classification_weights,
)

# --------- VISUALISATION ---------
# can plot what we have at this stage
plt.imshow(dot_prod_oneimage.reshape(10, 10))

# for better results, better to upsample
resized_dot_prod = tf.image.resize(
    dot_prod_oneimage, (299, 299), antialias=True
).numpy()

plt.figure(figsize=(10, 10))
plt.imshow(batch_test[0][selected_image_index])
plt.imshow(
    resized_dot_prod.reshape(299, 299), cmap="jet", alpha=0.3, interpolation="nearest"
)
plt.show()

