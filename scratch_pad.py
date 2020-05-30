# ---------------- not used anywhere at the moment -----------------


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


def normalize(input_image, input_mask):
  input_image = tf.cast(input_image, tf.float32) / 255.0
  input_mask -= 1
  return input_image, input_mask

@tf.function
def load_image_train(datapoint):
  input_image = tf.image.resize(datapoint['image'], (128, 128))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

  if tf.random.uniform(()) > 0.5:
    input_image = tf.image.flip_left_right(input_image)
    input_mask = tf.image.flip_left_right(input_mask)

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask

def load_image_test(datapoint):
input_image = tf.image.resize(datapoint['image'], (128, 128))
input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

input_image, input_mask = normalize(input_image, input_mask)

return input_image, input_mask

TRAIN_LENGTH = info.splits['train'].num_examples
BATCH_SIZE = 64
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

train = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test = dataset['test'].map(load_image_test)

train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset = test.batch(BATCH_SIZE)

def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]

EPOCHS = 20
VAL_SUBSPLITS = 5
VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS

model_history = model.fit(train_dataset, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test_dataset,
                          callbacks=[DisplayCallback()])

# -------------------- other idea --------------
# https://github.com/keras-team/keras/issues/3059

def train_generator(img_dir, label_dir, batch_size, input_size):
    list_images = os.listdir(img_dir)
    shuffle(list_images) #Randomize the choice of batches
    ids_train_split = range(len(list_images))
    while True:
         for start in range(0, len(ids_train_split), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(ids_train_split))
            ids_train_batch = ids_train_split[start:end]
            for id in ids_train_batch:
                img = cv2.imread(os.path.join(img_dir, list_images[id]))
                img = cv2.resize(img, (input_size[0], input_size[1]))
                mask = cv2.imread(os.path.join(label_dir, list_images[id].replace('jpg', 'png')), 0)
                mask = cv2.resize(mask, (input_size[0], input_size[1]))
                mask = np.expand_dims(mask, axis=2)
                x_batch.append(img)
                y_batch.append(mask)

            x_batch = np.array(x_batch, np.float32) / 255.
            y_batch = np.array(y_batch, np.float32)

            yield x_batch, y_batch