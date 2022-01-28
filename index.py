import tensorflow as tf

train_ds = tf.keras.utils.image_dataset_from_directory(
    'TRAIN',
    shuffle=True,
    image_size=(128, 128),
    color_mode='grayscale',
    validation_split=0.75,
    subset="training",
    seed=1337
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    'TRAIN',
    shuffle=True,
    image_size=(128, 128),
    color_mode='grayscale',
    validation_split=0.25,
    subset="validation",
    seed=1337
)
print(len(list(train_ds)))
for image, label in train_ds.take(1):
    print(label)