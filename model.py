import tensorflow as tf

def get_model():
    preprocessing_layers = [
        tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(80,80, 1))
    ]
    def conv_2d_pooling_layers(filters):
        return [
            tf.keras.layers.Conv2D(
                filters,
                (2,2),
                activation='relu',
                padding='same'
            ),
            tf.keras.layers.MaxPooling2D()
        ]
    core_layers = \
        conv_2d_pooling_layers(32) + \
        conv_2d_pooling_layers(64) + \
        conv_2d_pooling_layers(128)

    dense_layers = [
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(33)
    ]

    model = tf.keras.Sequential(
        preprocessing_layers +
        core_layers +
        dense_layers
    )


    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    model.compile(
        optimizer='adam',
        loss=loss,
        metrics=['accuracy']
    )

    return model