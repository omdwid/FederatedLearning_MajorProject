import tensorflow as tf
import numpy as np

# Load and normalize the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Shuffle and partition the training data
train_data = list(zip(x_train, y_train))
np.random.shuffle(train_data)
split_index = len(train_data) // 2
client1_train_data = train_data[:split_index]
client2_train_data = train_data[split_index:]

# Partition the test data similarly (optional: shuffle test data too)
test_data = list(zip(x_test, y_test))
np.random.shuffle(test_data)
split_index = len(test_data) // 2
client1_test_data = test_data[:split_index]
client2_test_data = test_data[split_index:]

# Unzip and save Client 1's data
x_train_client1, y_train_client1 = zip(*client1_train_data)
x_test_client1, y_test_client1 = zip(*client1_test_data)
np.savez(
    "client1_data.npz",
    x_train=np.array(x_train_client1),
    y_train=np.array(y_train_client1),
    x_test=np.array(x_test_client1),
    y_test=np.array(y_test_client1),
)

# Unzip and save Client 2's data
x_train_client2, y_train_client2 = zip(*client2_train_data)
x_test_client2, y_test_client2 = zip(*client2_test_data)
np.savez(
    "client2_data.npz",
    x_train=np.array(x_train_client2),
    y_train=np.array(y_train_client2),
    x_test=np.array(x_test_client2),
    y_test=np.array(y_test_client2),
)
