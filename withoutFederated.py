import flwr as fl
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# AUxillary methods
def getDist(y):
    ax = sns.countplot(y)
    ax.set(title="Count of data classes")
    plt.show()


def getData(dist, x, y):
    dx = []
    dy = []
    counts = [0 for i in range(10)]
    for i in range(len(x)):
        if counts[y[i]] < dist[y[i]]:
            dx.append(x[i])
            dy.append(y[i])
            counts[y[i]] += 1

    return np.array(dx), np.array(dy)


def generatePlot(x_train, y_train):
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(y=y_train)
    ax.set(
        title="Frequency of Classes in MNIST Dataset",
        xlabel="Digit Class",
        ylabel="Frequency",
    )
    plt.show()


# Load and compile Keras model
model = keras.Sequential(
    [
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ]
)
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[..., np.newaxis] / 255.0
generatePlot(x_train, y_train)


model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test), verbose=0)

loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print("Eval accuracy : ", accuracy)
