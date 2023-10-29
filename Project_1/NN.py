import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class NeuralNetwork:
    def __init__(self, layer_sizes=(4, 10, 3), learning_rate=0.3, regularization=0, epochs=2, batch_size=100):
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.epochs = epochs
        self.batch_size = batch_size

        self.layer_sizes = layer_sizes
        self.weights = [self.generate_weights(layer_sizes[i + 1], layer_sizes[i] + 1) for i in
                        range(len(layer_sizes) - 1)]

    """
        Randomly generate weight matrix.
    """

    def generate_weights(self, rows, columns):
        return np.random.rand(rows, columns) - 0.5

    """
        Sigmoid activation function.
    """

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    """
        Relu activation function.
    """

    def relu(self, x):
        return np.maximum(0, x)

    """
        Tanh activation function.
    """

    def tanh(self, x):
        return np.tanh(x)

    """
        Column-vectorize an array.
    """

    def column_vectorize(self, arr):
        return np.array(arr).reshape(len(arr), 1)

    """
        Row-vectorize an array.
    """

    def row_vectorize(self, arr):
        return np.array(arr).reshape(1, len(arr))

    """
        Vectorize the labels.
     """

    def vectorize_labels(self, labels):
        return [[int(labels[i] == j) for j in range(self.layer_sizes[-1])] for i in range(len(labels))]

    """
        Adds a bias unit to activations of a layer.
    """

    def add_bias_unit(self, x):
        return np.insert(x, 0, 1)

    """
        Forward propagation.
    """

    def forward_propagation(self, x):
        x = self.add_bias_unit(x)
        activations = [x]

        for weights in self.weights:
            z = np.matmul(weights, self.column_vectorize(x))
            x = self.add_bias_unit(self.sigmoid(z))
            activations.append(x)
        return activations

    """
        Backward propagation.
    """

    def backward_propagation(self, activations, labels):
        output = activations[-1][1:]

        deltas = [np.zeros(layer_size) for layer_size in self.layer_sizes]
        gradients = [[] for _ in range(len(self.layer_sizes) - 1)]

        deltas[-1] = (output - labels) * output * (1 - output)
        for index in range(len(deltas) - 1, 0, -1):
            i = index - 1
            deltas[i] = np.matmul(self.weights[i].T[1:], deltas[i + 1]) * activations[i][1:] * (1 - activations[i][1:])
            gradients[i] = np.matmul(self.column_vectorize(deltas[i + 1]), self.row_vectorize(activations[i]))

        return gradients

    """
        Fits the Neural Network's weights using backpropagation.
    """

    def fit(self, input_data, labels):
        input_data = np.array(input_data)
        labels = np.array(self.vectorize_labels(labels))

        for _ in range(self.epochs):
            for batch in range(len(input_data) // self.batch_size):
                start_ix = batch * self.batch_size
                end_ix = (batch + 1) * self.batch_size

                gradients = np.array(self.weights) * 0
                for i, x in enumerate(input_data[start_ix:end_ix]):
                    # Forward propagation
                    activations = self.forward_propagation(x)

                    # Backward propagation
                    new_gradients = self.backward_propagation(activations, labels[i])
                    for layer, gradient in zip(range(len(self.weights)), new_gradients):
                        gradients[layer] += gradient

                # Update weights
                for layer in range(len(self.weights)):
                    regularization_matrix = np.array(self.weights[layer]) * self.regularization
                    regularization_matrix.T[0] = 0
                    self.weights[layer] -= (self.learning_rate * gradients[
                        layer] + regularization_matrix) / self.batch_size

    """
        Calculate the Mean Squared Error.
    """

    def mean_squared_error(self, input_data, labels):
        sq_error = 0
        for i, x in enumerate(input_data):
            sq_error += np.sum(np.power(self.forward_propagation(x)[-1][1:] - labels[i], 2))
        return sq_error / len(input_data)

    """
        Predicts values for the inputs through forward propagation.
    """

    def predict(self, input_data):
        input_data = np.array(input_data)
        predictions = []
        for x in input_data:
            predictions.append(np.argmax(self.forward_propagation(x)[-1][1:]))
        return predictions


def load_data():
    # Load CSV file into a dataframe
    df = pd.read_csv("Project_1/iris.csv").sample(frac=1, random_state=123)
    # Execute any necessary transformations
    transform_features(df)
    # Split data into 60-40 ratio
    split = round(len(df) * .70)
    return (pd.DataFrame(df.iloc[:split, :]), pd.DataFrame(df.iloc[split:, :]))


def transform_features(df):
    # Convert labels from strings to numbers
    classes = list(df["CL"].unique())
    df["Y"] = df["CL"].map(lambda row: classes.index(row))


# Fetch data
train_df, test_df = load_data()

# Initialize network and features
nn = NeuralNetwork(layer_sizes=(4, 3), regularization=0, learning_rate=3, batch_size=90, epochs=1000)
features = ["SL", "SW", "PL", "PW"]

# Scale data
scaler = StandardScaler()
train_data = scaler.fit_transform(train_df[features].values)

# Fit the network
start_time = time.perf_counter()

nn.fit(train_data, list(train_df["Y"]))
end_time = time.perf_counter()
execution_time = end_time - start_time

# Make predictions
test_df["Predictions"] = nn.predict(scaler.transform(test_df[features]))
print(test_df)

# Calculate accuracy
accuracy = len(test_df[test_df["Predictions"] == test_df["Y"]]) / len(test_df)
print("The accuracy is: " + str(accuracy))
print("Execution time: ", round(execution_time, 3), "seconds")
