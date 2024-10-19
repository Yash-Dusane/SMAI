import numpy as np

class MLPRegression:
    def __init__(self, layers, learning_rate=0.01, activation='relu', optimizer='sgd', batch_size=32, epochs=100):
        self.layers = layers
        self.learning_rate = learning_rate
        self.activation_func = self.get_activation_func(activation)
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.weights, self.biases = self.initialize_weights()
    
    def initialize_weights(self):
        np.random.seed(42)
        weights = []
        biases = []
        for i in range(1, len(self.layers)):
            weights.append(np.random.randn(self.layers[i], self.layers[i - 1]) * 0.01)
            biases.append(np.zeros((self.layers[i], 1)))
        return weights, biases

    def get_activation_func(self, name):
        if name == 'sigmoid':
            return lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500))), lambda x: x * (1 - x)  # Clipping the input to avoid overflow
        elif name == 'tanh':
            return lambda x: np.tanh(x), lambda x: 1 - x ** 2  # Tanh & derivative
        elif name == 'relu':
            return lambda x: np.maximum(0, x), lambda x: np.where(x > 0, 1, 0)  # ReLU & derivative
        elif name == 'linear':
            return lambda x: x, lambda x: 1  # Linear (for regression)
    x: 1  # Linear (for regression)

    def forward(self, X):
        a = X.T
        activations = [a]
        z_values = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a = self.activation_func[0](z)
            z_values.append(z)
            activations.append(a)
        return activations, z_values

    def backprop(self, activations, z_values, y):
        m = y.shape[0]
        dz = activations[-1] - y.T
        dws = []
        dbs = []
        for i in reversed(range(len(self.weights))):
            dw = (1/m) * np.dot(dz, activations[i].T)
            db = (1/m) * np.sum(dz, axis=1, keepdims=True)

            # Gradient Clipping: Limit the values of dw and db to a certain threshold
            np.clip(dw, -1e5, 1e5, out=dw)
            np.clip(db, -1e5, 1e5, out=db)

            dws.insert(0, dw)
            dbs.insert(0, db)
            if i > 0:
                dz = np.dot(self.weights[i].T, dz) * self.activation_func[1](z_values[i-1])  # Backpropagate
        return dws, dbs


    def update_weights(self, dws, dbs):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dws[i]
            self.biases[i] -= self.learning_rate * dbs[i]

    def update_weights_batch(self, X_train, y_train):
 
        y_train = np.array(y_train).reshape(-1, 1)

        for epoch in range(self.epochs):
            activations, z_values = self.forward(X_train)
            dws, dbs = self.backprop(activations, z_values, y_train)
            self.update_weights(dws, dbs)


    def update_weights_mini_batch(self, X_train, y_train):
        for epoch in range(self.epochs):
            for start_idx in range(0, len(X_train), self.batch_size):
                end_idx = start_idx + self.batch_size
                X_batch = X_train[start_idx:end_idx]
                y_batch = np.array(y_train[start_idx:end_idx]).reshape(-1, 1)  # Ensure proper shape

                activations, z_values = self.forward(X_batch)
                dws, dbs = self.backprop(activations, z_values, y_batch)
                self.update_weights(dws, dbs)


    def fit(self, X_train, y_train):
        if self.optimizer == 'sgd':
            self.update_weights_batch(X_train, y_train)
        elif self.optimizer == 'batch':
            self.update_weights_batch(X_train, y_train)
        elif self.optimizer == 'mini_batch':
            self.update_weights_mini_batch(X_train, y_train)

    def predict(self, X):
        activations, _ = self.forward(X)
        return activations[-1].T

    def evaluate(self, X, y):
        y = np.array(y).reshape(-1, 1)
        y_pred = self.predict(X)

        # Ensure no divide by zero in loss calculation
        mae = np.mean((y - y_pred))
        mse = np.mean((y - y_pred) ** 2)
        if np.isnan(mse) or np.isinf(mse):
            print("MSE computation resulted in NaN or Inf.")
            mse = np.nan_to_num(mse, nan=1e6, posinf=1e6, neginf=-1e6)  # Avoid NaN or Inf by substituting large values
        rmse = np.sqrt(mse)
        r2 = 1 - (np.sum((y - y_pred) ** 2) / (np.sum((y - np.mean(y)) ** 2) + 1e-8))  # Small epsilon to avoid division by zero
        return mse, rmse, r2, mae


    def gradient_check(self, X, y, epsilon=1e-7):
        activations, _ = self.forward(X)
        dws, dbs = self.backprop(activations, None, y)
        
        # Gradient checking for weights
        for i in range(len(self.weights)):
            original_weights = self.weights[i].copy()
            num_grads_w = np.zeros_like(self.weights[i])
            
            for j in range(self.weights[i].shape[0]):
                for k in range(self.weights[i].shape[1]):
                    self.weights[i][j, k] += epsilon
                    loss1 = self._compute_loss(X, y)
                    
                    self.weights[i][j, k] -= 2 * epsilon
                    loss2 = self._compute_loss(X, y)
                    
                    num_grads_w[j, k] = (loss1 - loss2) / (2 * epsilon)
                    self.weights[i][j, k] = original_weights[j, k]  # Reset
            
            assert np.allclose(dws[i], num_grads_w, atol=1e-5), f"Gradient check failed for layer {i}"
        
        print("Gradient check passed for weights")

    def _compute_loss(self, X, y):
        y_pred = self.predict(X)
        return np.mean((y - y_pred) ** 2)

