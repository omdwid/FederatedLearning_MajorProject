import flwr as fl
import tensorflow as tf
from tensorflow import keras
import sys
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


def importData():
    # Load Client 1's data
    data = np.load("client1_data.npz")
    x_train_client1 = data["x_train"]
    y_train_client1 = data["y_train"]
    x_test_client1 = data["x_test"]
    y_test_client1 = data["y_test"]

    print("Client 1 train data shape:", x_train_client1.shape, y_train_client1.shape)
    print("Client 1 test data shape:", x_test_client1.shape, y_test_client1.shape)

    return (x_train_client1, y_train_client1), (x_test_client1, y_test_client1)


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

# Load dataset
# (x_train, y_train), (x_test, y_test) = importData()

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[..., np.newaxis] / 255.0
# dist = [4000, 4000, 4000, 3000, 10, 10, 10, 10, 4000, 10]
# x_train, y_train = getData(dist, x_train, y_train)
# getDist(y_train)
generatePlot(x_train, y_train)


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        r = model.fit(
            x_train, y_train, epochs=1, validation_data=(x_test, y_test), verbose=0
        )
        hist = r.history
        print("Fit history : ", hist)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        print("Eval accuracy : ", accuracy)
        return loss, len(x_test), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client(
    server_address="localhost:" + str(sys.argv[1]),
    client=FlowerClient(),
    grpc_max_message_length=1024 * 1024 * 1024,
)
