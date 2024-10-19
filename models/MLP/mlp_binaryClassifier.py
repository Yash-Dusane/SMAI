import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
import matplotlib.pyplot as plt

class MLPBinary:
    def _init_(self, input_size, learning_rate=0.01, loss_type='bce', epochs=100, activation='sigmoid', batch_size=32):
        self.learning_rate = learning_rate
        self.loss_type = loss_type
        self.epochs = epochs
        self.activation = activation
        self.batch_size = batch_size  # Added batch size for mini-batch/SGD
        self.weights = np.random.randn(input_size, 1) * 0.01
        self.bias = np.zeros((1, 1))
        self.losses = []

    # Activation functions
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def relu(self, z):
        return np.maximum(0, z)

    def tanh(self, z):
        return np.tanh(z)

    # Derivatives for activation functions (needed for backprop)
    def sigmoid_derivative(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def relu_derivative(self, z):
        return np.where(z > 0, 1, 0)

    def tanh_derivative(self, z):
        return 1 - np.tanh(z) ** 2

    # Forward pass based on selected activation function
    def forward(self, X):
        z = np.dot(X, self.weights) + self.bias
        if self.activation == 'sigmoid':
            return self.sigmoid(z)
        elif self.activation == 'relu':
            return self.relu(z)
        elif self.activation == 'tanh':
            return self.tanh(z)

    def compute_loss(self, A, y):
        m = y.shape[0]
        if self.loss_type == 'bce':
            loss = -np.mean(y * np.log(A) + (1 - y) * np.log(1 - A))
        elif self.loss_type == 'mse':
            loss = np.mean((A - y) ** 2)
        return loss

    def backprop(self, X, A, y):
        m = y.shape[0]
        dz = A - y  # Derivative of loss w.r.t activation output

        dw = (1/m) * np.dot(X.T, dz)
        db = (1/m) * np.sum(dz)
        return dw, db

    def update_weights(self, dw, db):
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

    def fit(self, X, y):
        y = y.reshape(-1, 1)
        m = X.shape[0]

        for epoch in range(self.epochs):
            permutation = np.random.permutation(m)  # Shuffle the data for mini-batch/SGD
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]

            for i in range(0, m, self.batch_size):
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]

                A = self.forward(X_batch)
                loss = self.compute_loss(A, y_batch)
                dw, db = self.backprop(X_batch, A, y_batch)
                self.update_weights(dw, db)

            self.losses.append(loss)

            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

    def predict(self, X):
        A = self.forward(X)
        return (A > 0.5).astype(int)



# # Load dataset
# data = pd.read_csv('diabetes.csv')

# X = data.iloc[:, :-1].values  # All columns except the last
# y = data.iloc[:, -1].values   # Last column


# X_train, X_testval, y_train, y_testval = train_test_split(X, y, test_size=0.2, random_state=42)
# X_val, X_test, y_val, y_test = train_test_split(X_testval, y_testval, test_size=0.5, random_state=42)

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_val = scaler.transform(X_val)
# X_test = scaler.transform(X_test)
# Now, you can test with different batch sizes

# # Model with BCE loss, sigmoid activation, and SGD (batch size = 1)
# model_sgd = MLPBinary(input_size=X_train.shape[1], learning_rate=0.01, loss_type='bce', epochs=100, activation='sigmoid', batch_size=1)
# model_sgd.fit(X_train, y_train)

# # Model with MSE loss, relu activation, and mini-batch gradient descent (batch size = 32)
# model_mini_batch = MLPBinary(input_size=X_train.shape[1], learning_rate=0.01, loss_type='mse', epochs=100, activation='relu', batch_size=32)
# model_mini_batch.fit(X_train, y_train)

# # Predict and evaluate
# y_pred_sgd = model_sgd.predict(X_test)
# y_pred_mini_batch = model_mini_batch.predict(X_test)

# accuracy_sgd = np.mean(y_pred_sgd.flatten() == y_test)
# accuracy_mini_batch = np.mean(y_pred_mini_batch.flatten() == y_test)

# print(f'Accuracy with SGD (Batch size 1): {accuracy_sgd:.2f}')
# print(f'Accuracy with Mini-Batch (Batch size 32): {accuracy_mini_batch:.2f}')

# # Plotting the losses
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.plot(model_sgd.losses, label='SGD (Batch size 1)')
# plt.title('Loss with SGD (Batch size 1)')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(model_mini_batch.losses, label='Mini-Batch (Batch size 32)', color='red')
# plt.title('Loss with Mini-Batch (Batch size 32)')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()

# plt.tight_layout()
# plt.show()










#  Initial code :

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import wandb
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

# class MLPBinaryClassifier:
#     def __init__(self, input_size, learning_rate=0.01, loss_type='bce', epochs=100):
#         self.learning_rate = learning_rate
#         self.loss_type = loss_type
#         self.epochs = epochs
#         self.weights = np.random.randn(input_size, 1) * 0.01
#         self.bias = np.zeros((1, 1))
#         self.losses = []

#     def sigmoid(self, z):
#         return 1 / (1 + np.exp(-z))

#     def forward(self, X):
#         z = np.dot(X, self.weights) + self.bias
#         return self.sigmoid(z)

#     def compute_loss(self, A, y):
#         m = y.shape[0]
#         if self.loss_type == 'bce':
#             loss = -1/m * (np.dot(y.T, np.log(A)) + np.dot((1 - y).T, np.log(1 - A)))
#         elif self.loss_type == 'mse':
#             loss = np.mean((A - y) ** 2)
#         return loss.item()  # Return as scalar

#     def backprop(self, X, A, y):
#         m = y.shape[0]
#         dz = A - y
#         dw = (1/m) * np.dot(X.T, dz)
#         db = (1/m) * np.sum(dz)
#         return dw, db

#     def update_weights(self, dw, db):
#         self.weights -= self.learning_rate * dw
#         self.bias -= self.learning_rate * db

#     def fit(self, X, y):
#         y = y.reshape(-1, 1)
#         for epoch in range(self.epochs):
#             A = self.forward(X)
#             loss = self.compute_loss(A, y)
#             self.losses.append(loss)
#             dw, db = self.backprop(X, A, y)
#             self.update_weights(dw, db)

#             if epoch % 10 == 0:
#                 print(f'Epoch {epoch}, Loss: {loss}')
#                 wandb.log({"loss": loss})  # Log to WandB

#     def predict(self, X):
#         A = self.forward(X)
#         return (A > 0.5).astype(int)