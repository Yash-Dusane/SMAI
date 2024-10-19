import numpy as np
import wandb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class MLP_hyperparam:
    def __init__(self, config):
        self.hidden_layers = config.hidden_layers
        self.learning_rate = config.learning_rate
        self.activation = config.activation
        self.optimizer = config.optimizer
        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.early_stopping_patience = config.early_stopping_patience
        self.weights = []
        self.biases = []

    def _initialize_parameters(self, input_size, output_size):
        layer_sizes = [input_size] + self.hidden_layers + [output_size]
        for i in range(1, len(layer_sizes)):
            self.weights.append(np.random.randn(layer_sizes[i-1], layer_sizes[i]) * 0.01)
            self.biases.append(np.zeros((1, layer_sizes[i])))

    def _activation_function(self, x):
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
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

    def fit(self, X_train, y_train, X_val, y_val):
        
        n_samples, n_features = X_train.shape
        
        # Adjust output values to be between 0 and 5 (for one-hot encoding)
        y_train_adjusted = y_train - 3  # Shift labels from [3-8] to [0-5]
        y_val_adjusted = y_val - 3
        
        # Initialize parameters
        n_classes = 6  # Since we have classes from 0 to 5 after adjustment
        self._initialize_parameters(n_features, n_classes)
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            if self.optimizer == 'sgd':
                indices = np.random.permutation(n_samples)
                X_shuffled = X_train[indices]
                y_shuffled = y_train_adjusted[indices]
                for i in range(n_samples):
                    dW, db = self._backpropagation(X_shuffled[i:i+1], np.eye(n_classes)[y_shuffled[i:i+1]])
                    self._update_parameters(dW, db)
            elif self.optimizer == 'batch':
                dW, db = self._backpropagation(X_train, np.eye(n_classes)[y_train_adjusted])
                self._update_parameters(dW, db)
            elif self.optimizer == 'mini_batch':
                for i in range(0, n_samples, self.batch_size):
                    batch_X = X_train[i:i+self.batch_size]
                    batch_y = y_train_adjusted[i:i+self.batch_size]
                    dW, db = self._backpropagation(batch_X, np.eye(n_classes)[batch_y])
                    self._update_parameters(dW, db)

            train_loss = self._compute_cost(X_train, np.eye(n_classes)[y_train_adjusted])
            val_loss = self._compute_cost(X_val, np.eye(n_classes)[y_val_adjusted])
            
            train_pred = self.predict(X_train)
            val_pred = self.predict(X_val)
            train_accuracy = self.accuracy(y_train, train_pred)
            val_accuracy = self.accuracy(y_val, val_pred)
            
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_accuracy": train_accuracy,
                "val_accuracy": val_accuracy
            })
            
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        # Compute final metrics
        final_pred = self.predict(X_val)
        final_metrics = {
            "val_accuracy": accuracy_score(y_val, final_pred),
            "val_f1": f1_score(y_val, final_pred, average='weighted'),
            "val_precision": precision_score(y_val, final_pred, average='weighted'),
            "val_recall": recall_score(y_val, final_pred, average='weighted')
        }
        wandb.log(final_metrics)

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
