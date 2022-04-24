import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Flatten, Dense, Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import SGD



import os


def build_model(batch_size):
    model = Sequential(
        [
            InputLayer(input_shape=(97, 1024, 1), batch_size=batch_size, name="Input_layer"),

            Conv2D(128, 3, activation="relu", name="Convlayer1", data_format='channels_last'),
            MaxPooling2D((2, 4), padding="valid", data_format='channels_last', name='Pool1'),
            # Dropout(0.2),

            Conv2D(384, 3, activation="relu", name="Convlayer2", data_format='channels_last'),
            MaxPooling2D((4, 5), padding="valid", data_format='channels_last', name='Pool2'),
            # Dropout(0.2),

            Conv2D(768, 3, activation="relu", name="Convlayer3", data_format='channels_last'),
            MaxPooling2D((3, 8), padding="valid", name='Pool3', data_format='channels_last'),
            # Dropout(0.2),

            Conv2D(768, 3, activation="relu", name="Convlayer4", data_format='channels_last'),
            MaxPooling2D((1, 3), padding="valid", name='Pool4', data_format='channels_last'),

            Flatten(name="flatten"),
            Dense(8, activation="sigmoid", name="Dense", trainable=True)
        ]
    )
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    # model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy", "categorical_accuracy"])
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics="accuracy")

    return model


def build_model2(batch_size):
    model2 = Sequential([

        InputLayer(input_shape=(97, 1024, 1), batch_size=batch_size, name="Input_layer"),

        Conv2D(128, 3, activation="relu", name="Convlayer1", data_format='channels_last'),
        Conv2D(128, 3, activation="relu", name="Convlayer2", data_format='channels_last'),
        MaxPooling2D((2, 8), padding="valid", data_format='channels_last', name='Pool1'),

        Conv2D(328, 3, activation="relu", name="Convlayer3", data_format='channels_last'),
        Conv2D(328, 3, activation="relu", name="Convlayer4", data_format='channels_last'),
        MaxPooling2D((4, 8), padding="valid", data_format='channels_last', name='Pool2'),

        Flatten(),

        Dense(100, activation='relu'),
        Dense(100, activation='relu'),
        Dense(8, activation='sigmoid'),

    ])

    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    ## Baisser la leraning rate si la loss diverge
    model2.compile(loss="categorical_crossentropy", optimizer=opt, metrics="accuracy")

    return model2


if __name__ == "__main__":
    # test model
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    model = build_model2(16)
    print("Following model was built:")
    print(model.summary())
