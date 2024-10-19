import numpy as np

class MLPClassification:
    def __init__(self, 
                 hidden_layers, 
                 learning_rate=0.01,
                 activation='sigmoid',
                 optimizer='sgd',
                 batch_size=32,
                 epochs=100,
                 early_stopping_patience=10):
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.activation = activation
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.weights = []
        self.biases = []

    def _initialize_parameters(self, input_size, output_size):
        layer_sizes = [input_size] + self.hidden_layers + [output_size]
        for i in range(1, len(layer_sizes)):
            self.weights.append(np.random.randn(layer_sizes[i-1], layer_sizes[i]) * 0.01)
            self.biases.append(np.zeros((1, layer_sizes[i])))

    def _activation_function(self, x):
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -709, 709)))
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'relu':
            return np.maximum(0, x)
        else:
            raise ValueError("Invalid activation function")

    def _activation_derivative(self, x):
        if self.activation == 'sigmoid':
            return x * (1 - x)
        elif self.activation == 'tanh':
            return 1 - np.power(x, 2)
        elif self.activation == 'relu':
            return np.where(x > 0, 1, 0)
        else:
            raise ValueError("Invalid activation function")

    def _forward_propagation(self, X):
        activations = [X]
        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            a = self._activation_function(z)
            activations.append(a)
        return activations

    def _backpropagation(self, X, y):
        m = X.shape[0]
        activations = self._forward_propagation(X)
        
        dW = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]
        
        delta = activations[-1] - y
        for l in range(len(self.weights) - 1, -1, -1):
            dW[l] = np.dot(activations[l].T, delta) / m
            db[l] = np.sum(delta, axis=0, keepdims=True) / m
            if l > 0:
                delta = np.dot(delta, self.weights[l].T) * self._activation_derivative(activations[l])
        
        return dW, db

    def _update_parameters(self, dW, db):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dW[i]
            self.biases[i] -= self.learning_rate * db[i]

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Adjust output values to be between 0 and 5 (for one-hot encoding)
        y_adjusted = y - 3  # Shift labels from [3-8] to [0-5]
        
        # Initialize parameters
        n_classes = 6  # Since we have classes from 0 to 5 after adjustment
        self._initialize_parameters(n_features, n_classes)
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            if self.optimizer == 'sgd':
                indices = np.random.permutation(n_samples)
                X_shuffled = X[indices]
                y_shuffled = y_adjusted[indices]
                for i in range(n_samples):
                    dW, db = self._backpropagation(X_shuffled[i:i+1], np.eye(n_classes)[y_shuffled[i:i+1]])
                    self._update_parameters(dW, db)

            elif self.optimizer == 'batch':
                dW, db = self._backpropagation(X, np.eye(n_classes)[y_adjusted])
                self._update_parameters(dW, db)

            elif self.optimizer == 'mini_batch':
                for i in range(0, n_samples, self.batch_size):
                    batch_X = X[i:i+self.batch_size]
                    batch_y = y_adjusted[i:i+self.batch_size]
                    dW, db = self._backpropagation(batch_X, np.eye(n_classes)[batch_y])
                    self._update_parameters(dW, db)

            loss = self._compute_cost(X, np.eye(n_classes)[y_adjusted])
            
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

    def predict(self, X):
        activations = self._forward_propagation(X)
        predicted_indices = np.argmax(activations[-1], axis=1)
        
        # Adjust back to original quality scores (3-8)
        return predicted_indices + 3

    def accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)

    def _compute_cost(self, X, y):
        activations = self._forward_propagation(X)
        m = X.shape[0]
        epsilon = 1e-15
        cost = -np.sum(y * np.log(activations[-1] + epsilon) + (1 - y) * np.log(1 - activations[-1] + epsilon)) / m
        return cost

    def gradient_check(self, X, y, epsilon=1e-7):
        y_adjusted = y - 3
        y_encoded = np.eye(6)[y_adjusted]
        
        dW, db = self._backpropagation(X, y_encoded)
        
        params = self.weights + self.biases
        grads = dW + db
        
        num_grads = []
        for param in params:
            num_grad = np.zeros_like(param)
            it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                idx = it.multi_index
                old_value = param[idx]
                
                param[idx] = old_value + epsilon
                cost_plus = self._compute_cost(X, y_encoded)
                
                param[idx] = old_value - epsilon
                cost_minus = self._compute_cost(X, y_encoded)
                
                num_grad[idx] = (cost_plus - cost_minus) / (2 * epsilon)
                
                param[idx] = old_value
                it.iternext()
            
            num_grads.append(num_grad)
        
        total_error = 0
        for grad, num_grad in zip(grads, num_grads):
            numerator = np.linalg.norm(grad - num_grad)
            denominator = np.linalg.norm(grad) + np.linalg.norm(num_grad)
            total_error += numerator / (denominator + 1e-7)
        
        average_error = total_error / len(params)
        print(f"Average relative error: {average_error}")
        
        return average_error < 1e-7