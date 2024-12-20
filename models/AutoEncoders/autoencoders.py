import numpy as np

class MLP:
    def __init__(self, layers, activation='relu', learning_rate=0.01, optimizer='sgd', batch_size=32, epochs=100):
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
            return lambda x: 1 / (1 + np.exp(-x)), lambda x: x * (1 - x)
        elif name == 'tanh':
            return lambda x: np.tanh(x), lambda x: 1 - x ** 2
        elif name == 'relu':
            return lambda x: np.maximum(0, x), lambda x: np.where(x > 0, 1, 0)
        elif name == 'linear':
            return lambda x: x, lambda x: 1

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
        dz = activations[-1] - y
        dws = []
        dbs = []
        for i in reversed(range(len(self.weights))):
            dw = (1/m) * np.dot(dz, activations[i].T)
            db = (1/m) * np.sum(dz, axis=1, keepdims=True)
            dws.insert(0, dw)
            dbs.insert(0, db)
            if i > 0:
                dz = np.dot(self.weights[i].T, dz) * self.activation_func[1](z_values[i-1])
        return dws, dbs

    def update_weights(self, dws, dbs):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dws[i]
            self.biases[i] -= self.learning_rate * dbs[i]

    def fit_sgd(self, X_train, y_train):
        y_train = y_train.T
        for epoch in range(self.epochs):
            for i in range(X_train.shape[0]):
                X_sample = X_train[i:i+1]
                y_sample = y_train[:, i:i+1]
                activations, z_values = self.forward(X_sample)
                dws, dbs = self.backprop(activations, z_values, y_sample)
                self.update_weights(dws, dbs)

    def fit_batch(self, X_train, y_train):
        y_train = y_train.T
        for epoch in range(self.epochs):
            activations, z_values = self.forward(X_train)
            dws, dbs = self.backprop(activations, z_values, y_train)
            self.update_weights(dws, dbs)

    def fit_mini_batch(self, X_train, y_train):
        y_train = y_train.T
        for epoch in range(self.epochs):
            for start_idx in range(0, X_train.shape[0], self.batch_size):
                end_idx = start_idx + self.batch_size
                X_batch = X_train[start_idx:end_idx]
                y_batch = y_train[:, start_idx:end_idx]
                activations, z_values = self.forward(X_batch)
                dws, dbs = self.backprop(activations, z_values, y_batch)
                self.update_weights(dws, dbs)

            # Calculate and print loss for the epoch
            # loss = np.mean((self.predict(X_train) - y_train) ** 2)
            # print(f'Epoch {epoch + 1}, Loss: {loss}')

    def fit(self, X_train, y_train):
        if self.optimizer == 'sgd':
            self.fit_sgd(X_train, y_train)
        elif self.optimizer == 'batch':
            self.fit_batch(X_train, y_train)
        elif self.optimizer == 'mini_batch':
            self.fit_mini_batch(X_train, y_train)

    def predict(self, X):
        activations, _ = self.forward(X)
        return activations[-1].T


class AutoEncoder:
    def __init__(self, input_size, hidden_layers, latent_size, activation='relu', optimizer='sgd', learning_rate=0.01, batch_size=32, epochs=100):
        
        # Define encoder structure: input_size -> hidden_layers -> latent_size
        encoder_layers = [input_size] + hidden_layers + [latent_size]
        self.encoder = MLP(encoder_layers, activation, learning_rate, optimizer, batch_size, epochs)

        # Define decoder structure: latent_size -> reversed(hidden_layers) -> input_size
        decoder_layers = [latent_size] + hidden_layers[::-1] + [input_size]
        self.decoder = MLP(decoder_layers, activation, learning_rate, optimizer, batch_size, epochs)

    def fit(self, X):
        
        X = np.array(X, dtype=float)
        assert X.ndim == 2, "Input X must be a 2D array (batch_size, input_size)."

        for epoch in range(self.encoder.epochs):
            encoded = self.encoder.predict(X)  # Shape: (batch_size, latent_size)

            decoded = self.decoder.predict(encoded)  # Shape: (batch_size, input_size)

            assert decoded.shape == X.shape, f"Shape mismatch: X.shape={X.shape}, decoded.shape={decoded.shape}"

            # Compute the loss (MSE)
            loss = np.mean((X - decoded) ** 2)
            print(f'Epoch {epoch + 1}, Loss: {loss}')
\
            self.encoder.fit(X, encoded)  # Train encoder
            self.decoder.fit(encoded, X)  # Train decoder

    def get_latent(self, X):
       
        X = np.array(X, dtype=float)
        return self.encoder.predict(X)

    def reconstruct(self, X):
        
        X = np.array(X, dtype=float)
        encoded = self.encoder.predict(X)
        return self.decoder.predict(encoded)
