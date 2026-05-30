# =========================================================
# Exercise 4A: Neural Network From Scratch
# =========================================================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# =========================================================
# Activation Functions
# =========================================================

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(a):
    return a * (1 - a)


def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return (z > 0).astype(float)


def tanh(z):
    return np.tanh(z)


def tanh_derivative(a):
    return 1 - a**2


# =========================================================
# Binary Cross Entropy Loss
# =========================================================

def binary_cross_entropy(y, y_hat):

    epsilon = 1e-8

    y_hat = np.clip(
        y_hat,
        epsilon,
        1 - epsilon
    )

    loss = -np.mean(
        y * np.log(y_hat)
        +
        (1 - y) * np.log(1 - y_hat)
    )

    return loss


# =========================================================
# Neural Network Class
# =========================================================

class NeuralNetwork:

    def __init__(
            self,
            input_size,
            hidden1,
            hidden2,
            output_size,
            activation='relu'
    ):

        self.activation_name = activation

        # Xavier Initialization

        self.W1 = np.random.randn(
            hidden1,
            input_size
        ) * np.sqrt(1 / input_size)

        self.b1 = np.zeros((hidden1, 1))

        self.W2 = np.random.randn(
            hidden2,
            hidden1
        ) * np.sqrt(1 / hidden1)

        self.b2 = np.zeros((hidden2, 1))

        self.W3 = np.random.randn(
            output_size,
            hidden2
        ) * np.sqrt(1 / hidden2)

        self.b3 = np.zeros((output_size, 1))

        self.loss_history = []

    # =====================================================
    # Activation Selector
    # =====================================================

    def activation(self, z):

        if self.activation_name == "sigmoid":
            return sigmoid(z)

        elif self.activation_name == "relu":
            return relu(z)

        elif self.activation_name == "tanh":
            return tanh(z)

        else:
            raise ValueError("Unknown activation function")

    def activation_derivative(self, z, a):

        if self.activation_name == "sigmoid":
            return sigmoid_derivative(a)

        elif self.activation_name == "relu":
            return relu_derivative(z)

        elif self.activation_name == "tanh":
            return tanh_derivative(a)

        else:
            raise ValueError("Unknown activation function")

    # =====================================================
    # Forward Pass
    # =====================================================

    def forward(self, X):

        self.Z1 = self.W1 @ X + self.b1
        self.A1 = self.activation(self.Z1)

        self.Z2 = self.W2 @ self.A1 + self.b2
        self.A2 = self.activation(self.Z2)

        self.Z3 = self.W3 @ self.A2 + self.b3

        # Output Layer
        self.A3 = sigmoid(self.Z3)

        return self.A3

    # =====================================================
    # Backpropagation
    # =====================================================

    def backward(self, X, y):

        m = X.shape[1]

        # Output Layer

        dZ3 = self.A3 - y

        dW3 = (1 / m) * (dZ3 @ self.A2.T)

        db3 = (1 / m) * np.sum(
            dZ3,
            axis=1,
            keepdims=True
        )

        # Hidden Layer 2

        dA2 = self.W3.T @ dZ3

        dZ2 = dA2 * self.activation_derivative(
            self.Z2,
            self.A2
        )

        dW2 = (1 / m) * (dZ2 @ self.A1.T)

        db2 = (1 / m) * np.sum(
            dZ2,
            axis=1,
            keepdims=True
        )

        # Hidden Layer 1

        dA1 = self.W2.T @ dZ2

        dZ1 = dA1 * self.activation_derivative(
            self.Z1,
            self.A1
        )

        dW1 = (1 / m) * (dZ1 @ X.T)

        db1 = (1 / m) * np.sum(
            dZ1,
            axis=1,
            keepdims=True
        )

        gradients = {

            "dW1": dW1,
            "db1": db1,

            "dW2": dW2,
            "db2": db2,

            "dW3": dW3,
            "db3": db3
        }

        return gradients

    # =====================================================
    # Update Parameters
    # =====================================================

    def update_parameters(
            self,
            gradients,
            learning_rate
    ):

        self.W1 -= learning_rate * gradients["dW1"]
        self.b1 -= learning_rate * gradients["db1"]

        self.W2 -= learning_rate * gradients["dW2"]
        self.b2 -= learning_rate * gradients["db2"]

        self.W3 -= learning_rate * gradients["dW3"]
        self.b3 -= learning_rate * gradients["db3"]

    # =====================================================
    # Mini Batch Gradient Descent
    # =====================================================

    def train(
            self,
            X,
            y,
            epochs=1000,
            learning_rate=0.01,
            batch_size=32
    ):

        m = X.shape[1]

        for epoch in range(epochs):

            permutation = np.random.permutation(m)

            X_shuffled = X[:, permutation]
            y_shuffled = y[:, permutation]

            for i in range(0, m, batch_size):

                X_batch = X_shuffled[:, i:i + batch_size]
                y_batch = y_shuffled[:, i:i + batch_size]

                self.forward(X_batch)

                gradients = self.backward(
                    X_batch,
                    y_batch
                )

                self.update_parameters(
                    gradients,
                    learning_rate
                )

            if epoch % 100 == 0:

                predictions = self.forward(X)

                loss = binary_cross_entropy(
                    y,
                    predictions
                )

                self.loss_history.append(loss)

                print(
                    f"Epoch {epoch:4d} | "
                    f"Loss = {loss:.6f}"
                )

    # =====================================================
    # Prediction
    # =====================================================

    def predict(self, X):

        probabilities = self.forward(X)

        return (probabilities > 0.5).astype(int)


# =========================================================
# Gradient Checking
# =========================================================

def gradient_check():

    print("\n" + "=" * 60)
    print("GRADIENT CHECKING")
    print("=" * 60)

    x = np.array([[0.5], [0.2]])
    y = np.array([[1]])

    nn = NeuralNetwork(
        input_size=2,
        hidden1=4,
        hidden2=3,
        output_size=1,
        activation='tanh'
    )

    nn.forward(x)

    grads = nn.backward(x, y)

    analytical = grads["dW3"][0, 0]

    epsilon = 1e-5

    original = nn.W3[0, 0]

    nn.W3[0, 0] = original + epsilon
    loss_plus = binary_cross_entropy(
        y,
        nn.forward(x)
    )

    nn.W3[0, 0] = original - epsilon
    loss_minus = binary_cross_entropy(
        y,
        nn.forward(x)
    )

    numerical = (
        loss_plus - loss_minus
    ) / (2 * epsilon)

    nn.W3[0, 0] = original

    difference = abs(
        analytical - numerical
    )

    print(
        f"Analytical Gradient : {analytical:.8f}"
    )

    print(
        f"Numerical Gradient  : {numerical:.8f}"
    )

    print(
        f"Difference          : {difference:.10f}"
    )


# =========================================================
# Create Dataset
# =========================================================

print("\nGenerating Dataset...")

X, y = make_moons(
    n_samples=1000,
    noise=0.20,
    random_state=42
)

scaler = StandardScaler()

X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

X_train = X_train.T
X_test = X_test.T

y_train = y_train.reshape(1, -1)
y_test = y_test.reshape(1, -1)


# =========================================================
# Train Models
# =========================================================

activations = [
    "sigmoid",
    "relu",
    "tanh"
]

results = {}

for activation in activations:

    print("\n" + "=" * 60)
    print(
        f"TRAINING USING "
        f"{activation.upper()}"
    )
    print("=" * 60)

    model = NeuralNetwork(
        input_size=2,
        hidden1=16,
        hidden2=8,
        output_size=1,
        activation=activation
    )

    model.train(
        X_train,
        y_train,
        epochs=1000,
        learning_rate=0.01,
        batch_size=32
    )

    predictions = model.predict(X_test)

    accuracy = np.mean(
        predictions == y_test
    )

    results[activation] = accuracy

    print(
        f"{activation.upper()} "
        f"Accuracy = {accuracy:.4f}"
    )


# =========================================================
# Compare Activations
# =========================================================

print("\n" + "=" * 60)
print("ACTIVATION COMPARISON")
print("=" * 60)

for name, acc in results.items():

    print(
        f"{name.upper():10s} "
        f": {acc:.4f}"
    )


# =========================================================
# Gradient Check
# =========================================================

gradient_check()


# =========================================================
# Train Best Model Again
# =========================================================

best_activation = max(
    results,
    key=results.get
)

print(
    f"\nBest Activation = "
    f"{best_activation.upper()}"
)

best_model = NeuralNetwork(
    input_size=2,
    hidden1=16,
    hidden2=8,
    output_size=1,
    activation=best_activation
)

best_model.train(
    X_train,
    y_train,
    epochs=1000,
    learning_rate=0.01,
    batch_size=32
)

plt.figure(figsize=(8, 5))

plt.plot(best_model.loss_history)

plt.title(
    f"Learning Curve ({best_activation.upper()})"
)

plt.xlabel(
    "Checkpoint (Every 100 Epochs)"
)

plt.ylabel("Loss")

plt.grid(True)

plt.show()


# =========================================================
# Final Accuracy
# =========================================================

final_predictions = best_model.predict(
    X_test
)

final_accuracy = np.mean(
    final_predictions == y_test
)

print("\n" + "=" * 60)
print("FINAL MODEL PERFORMANCE")
print("=" * 60)

print(
    f"Final Accuracy = "
    f"{final_accuracy:.4f}"
)

print("\nExercise 4A Completed Successfully.")
