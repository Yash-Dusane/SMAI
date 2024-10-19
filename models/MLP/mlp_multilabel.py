import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss

class ActivationFunction:
    @staticmethod
    def sigmoid(x, derivative=False):
        sig = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        return sig * (1 - sig) if derivative else sig

    @staticmethod
    def tanh(x, derivative=False):
        t = np.tanh(x)
        return 1 - t**2 if derivative else t

    @staticmethod
    def relu(x, derivative=False):
        return np.where(x > 0, 1, 0) if derivative else np.maximum(0, x)

    @staticmethod
    def linear(x, derivative=False):
        return np.ones_like(x) if derivative else x

class Layer:
    def __init__(self, input_size, output_size, activation='relu'):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size) 
        self.bias = np.zeros((1, output_size))
        self.activation = getattr(ActivationFunction, activation)
        self.dW = np.zeros_like(self.weights)  
        self.db = np.zeros_like(self.bias)  

    def forward(self, inputs):
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.bias
        return self.activation(self.z)

    def backward(self, delta, learning_rate, optimizer):
        delta = delta * self.activation(self.z, derivative=True)
        self.dW = np.dot(self.inputs.T, delta)
        self.db = np.sum(delta, axis=0, keepdims=True)
        
        if optimizer == 'sgd':
            self.weights -= learning_rate * self.dW
            self.bias -= learning_rate * self.db
        elif optimizer == 'batch':
            return self.dW, self.db
        
        return np.dot(delta, self.weights.T)


class AdvancedMultiLabelMLP:
    def _init_(self, input_size, hidden_sizes, output_size, learning_rate=0.001, 
                 activation='relu', optimizer='sgd', batch_size=32, epochs=100, early_stopping_pat=5):
        self.layers = []
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_pat
        
        # Input layer
        self.layers.append(Layer(input_size, hidden_sizes[0], activation))
        
        # Hidden layers
        for i in range(1, len(hidden_sizes)):
            self.layers.append(Layer(hidden_sizes[i-1], hidden_sizes[i], activation))
        
        # Output layer
        self.layers.append(Layer(hidden_sizes[-1], output_size, 'sigmoid'))

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, X, y, output):
        delta = output - y
        gradients = []
        for layer in reversed(self.layers):
            delta, grad = layer.backward(delta, self.learning_rate, self.optimizer)
            gradients.append(grad)
        return gradients

    def fit(self, X, y, validation_data=None):
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.epochs):
            # Shuffle for stochastic and mini-batch gradient descent
            indices = np.random.permutation(len(X))
            X = X[indices]
            y = y[indices]
            
            # Different optimizers
            if self.optimizer == 'sgd':
                for i in range(len(X)):
                    batch_X = X[i:i+1]  # Single example for SGD
                    batch_y = y[i:i+1]
                    
                    output = self.forward(batch_X)
                    self.backward(batch_X, batch_y, output)
                    
            elif self.optimizer == 'mini-batch':
                for i in range(0, len(X), self.batch_size):
                    batch_X = X[i:i+self.batch_size]
                    batch_y = y[i:i+self.batch_size]
                    
                    output = self.forward(batch_X)
                    self.backward(batch_X, batch_y, output)
                    
            elif self.optimizer == 'batch':
                output = self.forward(X)
                self.backward(X, y, output)
            
            # Early stopping
            if validation_data is not None:
                val_X, val_y = validation_data
                val_output = self.predict(val_X)
                val_loss = self.compute_loss(val_y, val_output)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= self.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            if (epoch + 1) % 10 == 0:
                loss = self.compute_loss(y, self.predict(X))
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss:.4f}")

    def predict(self, X):
        return self.forward(X)

    def predict_binary(self, X, threshold=0.5):
        probabilities = self.predict(X)
        return (probabilities >= threshold).astype(int)

    def compute_loss(self, y_true, y_pred):
        return -np.mean(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))

    def gradient_check(self, X, y, epsilon=1e-7):

        output = self.forward(X)
        self.backward(X, y, output)

        params = []
        grads = []
        for layer in self.layers:
            params.append(layer.weights)
            params.append(layer.bias)
            grads.append(layer.dW)
            grads.append(layer.db)
        
        num_grads = []
        
        for param in params:
            num_grad = np.zeros_like(param)
            it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                idx = it.multi_index
                old_value = param[idx]

                param[idx] = old_value + epsilon
                cost_plus = self.compute_loss(y, self.forward(X))
 
                param[idx] = old_value - epsilon
                cost_minus = self.compute_loss(y, self.forward(X))
      
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






# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss

# class ActivationFunction:
#     @staticmethod
#     def sigmoid(x, derivative=False):
#         sig = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
#         return sig * (1 - sig) if derivative else sig

#     @staticmethod
#     def tanh(x, derivative=False):
#         t = np.tanh(x)
#         return 1 - t**2 if derivative else t

#     @staticmethod
#     def relu(x, derivative=False):
#         return np.where(x > 0, 1, 0) if derivative else np.maximum(0, x)

#     @staticmethod
#     def linear(x, derivative=False):
#         return np.ones_like(x) if derivative else x

# class Layer:
#     def __init__(self, input_size, output_size, activation='relu'):
#         self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size) 
#         self.bias = np.zeros((1, output_size))
#         self.activation = getattr(ActivationFunction, activation)
#         self.dW = np.zeros_like(self.weights)  
#         self.db = np.zeros_like(self.bias)  

#     def forward(self, inputs):
#         self.inputs = inputs
#         self.z = np.dot(inputs, self.weights) + self.bias
#         return self.activation(self.z)

#     def backward(self, delta, learning_rate, optimizer):
#         delta = delta * self.activation(self.z, derivative=True)
#         self.dW = np.dot(self.inputs.T, delta)
#         self.db = np.sum(delta, axis=0, keepdims=True)
        
#         if optimizer == 'sgd':
#             self.weights -= learning_rate * self.dW
#             self.bias -= learning_rate * self.db
#         elif optimizer == 'batch':
#             return self.dW, self.db
        
#         return np.dot(delta, self.weights.T)

# class AdvancedMultiLabelMLP:
#     def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.001, 
#                  activation='relu', optimizer='sgd', batch_size=32, epochs=100, early_stopping_pat=5):
#         self.layers = []
#         self.learning_rate = learning_rate
#         self.optimizer = optimizer
#         self.batch_size = batch_size
#         self.epochs = epochs
#         self.early_stopping_patience= early_stopping_pat

#         self.layers.append(Layer(input_size, hidden_sizes[0], activation))

#         for i in range(1, len(hidden_sizes)):
#             self.layers.append(Layer(hidden_sizes[i-1], hidden_sizes[i], activation))
        
#         self.layers.append(Layer(hidden_sizes[-1], output_size, 'sigmoid'))

#     def forward(self, X):
#         for layer in self.layers:
#             X = layer.forward(X)
#         return X

#     def backward(self, X, y, output):
#         delta = output - y
#         for layer in reversed(self.layers):
#             delta = layer.backward(delta, self.learning_rate, self.optimizer)

#     def fit(self, X, y, validation_data=None):
#         best_val_loss = float('inf')
#         patience_counter = 0

#         for epoch in range(self.epochs):
#             if self.optimizer == 'sgd':
#                 indices = np.random.permutation(len(X))
#                 X = X[indices]
#                 y = y[indices]
            
#             for i in range(0, len(X), self.batch_size):
#                 batch_X = X[i:i+self.batch_size]
#                 batch_y = y[i:i+self.batch_size]
                
#                 output = self.forward(batch_X)
#                 self.backward(batch_X, batch_y, output)
            
      
#             if validation_data is not None:
#                 val_X, val_y = validation_data
#                 val_output = self.predict(val_X)
#                 val_loss = self.compute_loss(val_y, val_output)
                
#                 if val_loss < best_val_loss:
#                     best_val_loss = val_loss
#                     patience_counter = 0
#                 else:
#                     patience_counter += 1
                
#                 if patience_counter >= self.early_stopping_patience:
#                     print(f"Early stopping at epoch {epoch+1}")
#                     break
            
#             if (epoch + 1) % 10 == 0:
#                 loss = self.compute_loss(y, self.predict(X))
#                 print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss:.4f}")

#     def predict(self, X):
#         return self.forward(X)

#     def predict_binary(self, X, threshold=0.5):
#         probabilities = self.predict(X)
#         return (probabilities >= threshold).astype(int)

#     def compute_loss(self, y_true, y_pred):
#         return -np.mean(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))

#     def gradient_check(self, X, y, epsilon=1e-7):

#         output = self.forward(X)
#         self.backward(X, y, output)

#         params = []
#         grads = []
#         for layer in self.layers:
#             params.append(layer.weights)
#             params.append(layer.bias)
#             grads.append(layer.dW)
#             grads.append(layer.db)
        
#         num_grads = []
        
#         for param in params:
#             num_grad = np.zeros_like(param)
#             it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
#             while not it.finished:
#                 idx = it.multi_index
#                 old_value = param[idx]

#                 param[idx] = old_value + epsilon
#                 cost_plus = self.compute_loss(y, self.forward(X))
 
#                 param[idx] = old_value - epsilon
#                 cost_minus = self.compute_loss(y, self.forward(X))
      
#                 num_grad[idx] = (cost_plus - cost_minus) / (2 * epsilon)
                
#                 param[idx] = old_value
#                 it.iternext()
            
#             num_grads.append(num_grad)
        
#         total_error = 0
#         for grad, num_grad in zip(grads, num_grads):
#             numerator = np.linalg.norm(grad - num_grad)
#             denominator = np.linalg.norm(grad) + np.linalg.norm(num_grad)
#             total_error += numerator / (denominator + 1e-7)
        
#         average_error = total_error / len(params)
#         print(f"Average relative error: {average_error}")
        
#         return average_error < 1e-7

    
# def preprocess_data(file_path):
#     df = pd.read_csv(file_path)
#     X = df.iloc[:, :-1]
#     y = df.iloc[:, -1].str.split()
    
#     le = LabelEncoder()
#     for column in X.columns:
#         if X[column].dtype == 'object':
#             X[column] = le.fit_transform(X[column])
    
#     scaler = StandardScaler()
#     X = scaler.fit_transform(X)
    
#     unique_labels = set([label for labels in y for label in labels])
#     label_to_index = {label: i for i, label in enumerate(unique_labels)}
#     y_encoded = np.zeros((len(y), len(unique_labels)))
#     for i, labels in enumerate(y):
#         for label in labels:
#             y_encoded[i, label_to_index[label]] = 1
    
#     return X, y_encoded, label_to_index

# def evaluate_model(y_true, y_pred):
#     accuracy = accuracy_score(y_true, y_pred)
#     precision = precision_score(y_true, y_pred, average='micro')
#     recall = recall_score(y_true, y_pred, average='micro')
#     f1 = f1_score(y_true, y_pred, average='micro')
#     h_loss = hamming_loss(y_true, y_pred)
    
#     print(f'Accuracy: {accuracy:.4f}')
#     print(f'Precision: {precision:.4f}')
#     print(f'Recall: {recall:.4f}')
#     print(f'F1-score: {f1:.4f}')
#     print(f'Hamming Loss: {h_loss:.4f}')

# def calculate_soft_accuracy(y_true, y_pred):
#     true_positives = np.sum((y_true == 1) & (y_pred == 1))
#     true_negatives = np.sum((y_true == 0) & (y_pred == 0))
    
#     total = y_true.shape[0] * y_true.shape[1]  

#     soft_accuracy = (true_positives + true_negatives) / total
#     return soft_accuracy
