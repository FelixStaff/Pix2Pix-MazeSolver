import tensorflow as tf
import numpy as np
import pickle

# first a function that load the data of Mazes
def load_obj(name):
    with open(name + ".pkl", "rb") as f:
        return pickle.load(f)


# create a class that will predict the solution of the maze with neural network with
# tensorflow 2.0 and keras and numpy
# the class will have a function that will create the neural network using keras.models
# the input size is the size of the maze with is (rows, cols, 1)
# if will be a convolutional neural network with 2 convolutional layers and 2 max pooling layers


def Net(input_size):
    return tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=3,
                padding="same",
                activation="relu",
                input_shape=input_size,
            ),
            tf.keras.layers.MaxPool2D(pool_size=2),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, padding="same", activation="relu"
            ),
            tf.keras.layers.MaxPool2D(pool_size=2),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Conv2DTranspose(
                filters=64, strides=2, kernel_size=3, padding="same", activation="relu"
            ),
            tf.keras.layers.Conv2DTranspose(
                filters=1, strides=2, kernel_size=3, padding="same", activation="relu"
            ),
        ]
    )


# train the model using gradient descent with GradientTape

# define a map function that convert a numpy array of float to a numpy array of int
# change the strings values following the next rules:
# ['.'] to [0], [#] to [1], [S] to [2], [E] to [3]
# and make it a new numpy array of float x_float
def map_func(x):
    x_float = np.zeros((x.shape[0], x.shape[1], 1))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i][j] == ".":
                x_float[i][j] = 0
            elif x[i][j] == "#":
                x_float[i][j] = 1
            elif x[i][j] == "S":
                x_float[i][j] = 2
            elif x[i][j] == "E":
                x_float[i][j] = 3
    return x_float


# create the model with input size (rows, cols, 1)
if __name__ == "__main__":
    # load the data of the mazes
    mazes = load_obj("Mazes")
    # split the (100, rows, cols, 2) list into (100, rows, cols) and (100, rows, cols)
    x_train = mazes[0]
    y_train = mazes[1]
    # make a new numpy array that saves the maze [n] as a numpy array of float
    for i in range(len(x_train)):
        x_train[i] = map_func(x_train[i])
    for i in range(len(y_train)):
        y_train[i] = map_func(y_train[i])
    x = tf.convert_to_tensor(x_train, dtype=tf.float32)
    y = tf.convert_to_tensor(y_train, dtype=tf.float32)
    print(x.shape)
    print(y.shape)
    print(x.dtype)
    # create the model
    model = Net((x.shape[1], x.shape[2], 1))
    model.compile(optimizer="adam", loss="mse")
    model.summary()
    # train the model
    with tf.device("/device:GPU:0"):
        model.fit(x, y, epochs=10, batch_size=32)
