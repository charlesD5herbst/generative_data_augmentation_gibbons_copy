import tensorflow as tf

class CNNNetwork:

    def custom_model(self):
        inputs = tf.keras.Input(shape=(128, 128, 1))
        x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2),
                                   padding='same', activation='relu')(inputs)
        x = tf.keras.layers.Dropout(rate=0.3)(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
        x = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=(2, 2),
                                   padding='same', activation='relu')(x)
        x = tf.keras.layers.Dropout(rate=0.2)(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(units=8, activation="relu")(x)
        x = tf.keras.layers.Dropout(rate=0.1)(x)

        outputs = tf.keras.layers.Dense(2, activation="softmax")(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="my_model")
        model.compile(optimizer="Adam", loss="binary_crossentropy",
                      metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

        return model

    def mobile_net(self):
        mobile_net = tf.keras.models.load_model('/Users/charlherbst/generative_data_augmentation/models/mobilenetv2_model.h5')
        for layer in mobile_net.layers:
            layer.trainable = False

        inputs = tf.keras.Input(shape=(128, 128, 3))
        x = mobile_net(inputs)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(units=8, activation="relu")(x)
        x = tf.keras.layers.Dropout(rate=0.2)(x)
        outputs = tf.keras.layers.Dense(2 , activation="softmax")(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="my_model")
        model.compile(optimizer="Adam", loss="binary_crossentropy",
                      metrics=["accuracy", tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
        return model