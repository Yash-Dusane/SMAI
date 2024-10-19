import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import wandb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models')))

from MLP.mlp_classification import MLPClassification
from MLP.mlp_classify_wandb import MLP_hyperparam
from MLP.mlp_multilabel import AdvancedMultiLabelMLP
from MLP.mlp_regression import MLPRegression
from MLP.mlp_binaryClassifier import MLPBinaryClassifier

from AutoEncoders.autoencoders import AutoEncoder



# -------------------- MLP CLassification -----------------------------

# Load the dataset
data = pd.read_csv('../../data/external/WineQT.csv')

# Describe the dataset
description = data.describe()
print(description)

# Plotting the distribution of wine quality
plt.figure(figsize=(10, 6))
data['quality'].value_counts().sort_index().plot(kind='bar')
plt.title('Distribution of Wine Quality')
plt.xlabel('Quality')
plt.ylabel('Frequency')
plt.xticks(rotation=0)
plt.show()


# Separate features and target variable
X = data.drop('quality', axis=1).values
y = data['quality'].values

# Normalize and standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_val_test, y_train, y_val_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)

mlp=MLPClassification(hidden_layers=[9],
        learning_rate=0.01,
        activation='sigmoid',
        optimizer='sgd',batch_size=32,
        epochs=1000)

# Fit the model on training data.
mlp.fit(X_train,y_train)

# Make predictions on the test set.
predictions=mlp.predict(X_val)

# Calculate accuracy.
accuracy_score=mlp.accuracy(y_val,predictions)
print(f"Accuracy: {accuracy_score * 100:.2f}%")

# Perform gradient checking on a sample of test data.
is_gradient_correct=mlp.gradient_check(X_val,y_val)
print(f"Gradient check passed: {is_gradient_correct}")

# ----------hyperpparam :

   
def train():

    config_defaults = {
        'hidden_layers': [64],
        'learning_rate': 0.01,
        'activation': 'sigmoid',
        'optimizer': 'sgd',
        'batch_size': 32,
        'epochs': 100,
        'early_stopping_patience': 10
    }

    wandb.init(config=config_defaults)
    config = wandb.config

    model = MLP_hyperparam(config)
    model.fit(X_train, y_train, X_val, y_val)

sweep_configuration = {
    'method': 'random',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'val_accuracy'},
    'parameters': 
    {
        'hidden_layers': {'values': [[9], [10, 8], [64, 32]]},
        'learning_rate': {'min': 0.0001, 'max': 0.1},
        'activation': {'values': ['sigmoid', 'relu', 'tanh']},
        'optimizer': {'values': ['sgd', 'batch', 'mini_batch']},
        'batch_size': {'values': [16, 32, 64]},
        'epochs': {'values': [50, 100, 150]},
        'early_stopping_patience': {'values': [5, 10, 15]}
    }
}

sweep_id = wandb.sweep(sweep_configuration, project="MLP_Hyperparameter_Tuning")
wandb.agent(sweep_id, train, count=20)  # Run 20 experiments

# After the sweep is complete, you can get the best run:
api = wandb.Api()
sweep = api.sweep(f"your_username/MLP_Hyperparameter_Tuning/{sweep_id}")
best_run = sweep.best_run()
print(f"Best run id: {best_run.id}")
print(f"Best run config: {best_run.config}")
print(f"Best run metrics: {best_run.summary}")


# ---------------- Running best model

mlp=MLPClassification(hidden_layers=[64, 32],
            learning_rate=0.03026,
            activation='relu',
            optimizer='sgd',batch_size=64,
            epochs=50)

# Fit the model on training data.
mlp.fit(X_train,y_train)

# Make predictions on the test set.
predictions=mlp.predict(X_val)

# Calculate accuracy.
accuracy_score=mlp.accuracy(y_val,predictions)
print(f"Accuracy: {accuracy_score * 100:.2f}%")

# Perform gradient checking on a sample of test data.
is_gradient_correct=mlp.gradient_check(X_val,y_val)
print(f"Gradient check passed: {is_gradient_correct}")



# ---------------------- Analysis of effects

def experiment_activation_functions(X_train, y_train, X_val, y_val):
    activation_functions = ['sigmoid', 'tanh', 'relu', 'linear']
    losses = []

    for activation in activation_functions:
        mlp=MLPClassification(hidden_layers=[64, 32],
            learning_rate=0.03026,
            activation=activation,
            optimizer='sgd',batch_size=64,
            epochs=50)
        mlp.fit(X_train, y_train)
        losses.append(mlp.losses)  # Assuming you store losses in the fit method

    # Plotting
    plt.figure(figsize=(10, 6))
    for i, loss in enumerate(losses):
        plt.plot(loss, label=activation_functions[i])
    
    plt.title('Effect of Non-linearity on Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def experiment_learning_rates(X_train, y_train, X_val, y_val):
    learning_rates = [0.001, 0.01, 0.1, 0.5]
    losses = []

    for lr in learning_rates:
        mlp=MLPClassification(hidden_layers=[64, 32],
            learning_rate=learning_rates,
            activation='relu',
            optimizer='sgd',batch_size=64,
            epochs=50)
        mlp.fit(X_train, y_train)
        losses.append(mlp.losses)

    # Plotting
    plt.figure(figsize=(10, 6))
    for i, loss in enumerate(losses):
        plt.plot(loss, label=f'LR: {learning_rates[i]}')
    
    plt.title('Effect of Learning Rate on Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def experiment_batch_sizes(X_train, y_train, X_val, y_val):
    batch_sizes = [1, 16, 32, 64]
    losses = []

    for batch_size in batch_sizes:
        mlp=MLPClassification(hidden_layers=[64, 32],
            learning_rate=0.03026,
            activation='relu',
            optimizer='sgd',batch_size=batch_size,
            epochs=50)
        mlp.fit(X_train, y_train)
        losses.append(mlp.losses)

    # Plotting
    plt.figure(figsize=(10, 6))
    for i, loss in enumerate(losses):
        plt.plot(loss, label=f'Batch Size: {batch_sizes[i]}')
    
    plt.title('Effect of Batch Size on Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

experiment_activation_functions(X_train, y_train, X_val, y_val)
experiment_learning_rates(X_train, y_train, X_val, y_val)
experiment_batch_sizes(X_train, y_train, X_val, y_val)





# --------------------- Multilabel


from sklearn.preprocessing import StandardScaler, LabelEncoder

# Step 1: Load the data
file_path = '../../data/external/advertisement.csv'  # Replace with your CSV file path
data = pd.read_csv(file_path)

# Step 2: Process the labels
# Assuming the last column contains labels in a string format like "label1 label2 label3"
labels = data.iloc[:, -1].str.get_dummies(sep=' ')  # Convert to one-hot encoding

# Step 3: Count label frequencies
label_counts = labels.sum().sort_values(ascending=False)

# Step 4: Plot the distribution
plt.figure(figsize=(12, 6))
plt.bar(label_counts.index, label_counts.values, color='blue')  # Use a single color like 'blue'
plt.xticks(rotation=45)
plt.title('Label Distribution in Multilabel Classification')
plt.xlabel('Labels')
plt.ylabel('Frequency')
plt.tight_layout()  # Adjust layout to fit labels
plt.show()


X = data.iloc[:, :-1].values  # Features
y = labels.values  # One-hot encoding for labels

X = data.iloc[:, :-1]
y = data.iloc[:, -1].str.split()

le = LabelEncoder()
for column in X.columns:
    if X[column].dtype == 'object':
        X[column] = le.fit_transform(X[column])

scaler = StandardScaler()
X = scaler.fit_transform(X)

unique_labels = set([label for labels in y for label in labels])
label_to_index = {label: i for i, label in enumerate(unique_labels)}
y_encoded = np.zeros((len(y), len(unique_labels)))
for i, labels in enumerate(y):
    for label in labels:
        y_encoded[i, label_to_index[label]] = 1


X_train, X_temp, y_train, y_temp = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


# MLP

input_size = X_train.shape[1]
hidden_sizes = [9]  # You can modify this to change the number of hidden layers and neurons
output_size = y_train.shape[1]

model = AdvancedMultiLabelMLP(
    input_size=input_size,
    hidden_sizes=hidden_sizes,
    output_size=output_size,
    learning_rate=0.01,
    activation='sigmoid',  # You can change this to 'sigmoid', 'tanh', or 'linear'
    optimizer='sgd',    # You can change this to 'batch' or 'mini-batch'
    batch_size=32,
    epochs=100,
    early_stopping_pat=5
)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss

def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='micro')
    recall = recall_score(y_true, y_pred, average='micro')
    f1 = f1_score(y_true, y_pred, average='micro')
    h_loss = hamming_loss(y_true, y_pred)
    
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-score: {f1:.4f}')
    print(f'Hamming Loss: {h_loss:.4f}')

def calculate_soft_accuracy(y_true, y_pred):
    # Count true positives and true negatives
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    true_negatives = np.sum((y_true == 0) & (y_pred == 0))
    
    total = y_true.shape[0] * y_true.shape[1]  # Total number of predictions
    
    # Calculate soft accuracy
    soft_accuracy = (true_positives + true_negatives) / total
    return soft_accuracy


# Train the model with early stopping
model.fit(X_train, y_train, validation_data=(X_test, y_test))

# Evaluate the model
y_pred_binary = model.predict_binary(X_test)
evaluate_model(y_test, y_pred_binary)

soft_accuracy = calculate_soft_accuracy(y_test, y_pred_binary)
print(f"\nSoft Accuracy: {soft_accuracy:.4f}")


# wandb

def run_experiment(config=None):
    with wandb.init(config=config):
        config = wandb.config
        
        input_size = X_train.shape[1]
        output_size = y_train.shape[1]
 
        model = AdvancedMultiLabelMLP(
            input_size=input_size,
            hidden_sizes=config.hidden_layers,
            output_size=output_size,
            learning_rate=config.learning_rate,
            activation=config.activation,
            optimizer=config.optimizer,
            batch_size=config.batch_size,
            epochs=config.epochs
        )

        model.fit(X_train, y_train, validation_data=(X_test, y_test), early_stopping_patience=config.early_stopping_patience)

        y_pred_binary = model.predict_binary(X_test)
        accuracy = accuracy_score(y_test, y_pred_binary)
        soft_accuracy = calculate_soft_accuracy(y_test, y_pred_binary)

        wandb.log({
            "val_accuracy": accuracy,
            "soft_accuracy": soft_accuracy
        })

        print(f"Accuracy: {accuracy:.4f}, Soft Accuracy: {soft_accuracy:.4f}")

sweep_configuration = {
    'method': 'random',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'soft_accuracy'},
    'parameters': 
    {
        'hidden_layers': {'values': [[9], [10, 8], [64, 32]]},
        'learning_rate': {'min': 0.0001, 'max': 0.1},
        'activation': {'values': ['sigmoid', 'relu', 'tanh']},
        'optimizer': {'values': ['sgd', 'batch', 'mini_batch']},
        'batch_size': {'values': [16, 32, 64]},
        'epochs': {'values': [50, 100, 150]},
        'early_stopping_patience': {'values': [5, 10, 15]}
    }
}
sweep_id = wandb.sweep(sweep_configuration, project="multi_label_mlp")

wandb.agent(sweep_id, function=run_experiment)

# ------------- Best Model

model = AdvancedMultiLabelMLP(
        input_size=input_size,
        hidden_sizes=[9],
        output_size=output_size,
        learning_rate=0.01234,
        activation='relu',
        optimizer='sgd',  
        batch_size=16,
        epochs=50,
        early_stopping_pat=5
    )
    
# Train the model with early stopping
model.fit(X_train, y_train, validation_data=(X_test, y_test))

# Evaluate the model
y_pred_binary = model.predict_binary(X_test)
evaluate_model(y_test, y_pred_binary)

soft_accuracy = calculate_soft_accuracy(y_test, y_pred_binary)
print(f"\nSoft Accuracy: {soft_accuracy:.4f}")



#  -----------------------------------    MLP Regression --------------------------------------------------------------


import pandas as pd

# Load the dataset from CSV
df = pd.read_csv('../../data/external/HousingData.csv')

# Describe the dataset
description = df.describe()
print(description)

# Replace 'NA' with NaN
df.replace('NA', pd.NA, inplace=True)

# Convert relevant columns to numeric (if they aren't already)
df = df.apply(pd.to_numeric, errors='coerce')

# Check for missing values
missing_values = df.isnull().sum()
print("Missing values in each column:\n", missing_values)

# Drop rows with missing values
df = df.dropna()

# Check again for missing values
missing_values_after = df.isnull().sum()
print("Missing values after handling:\n", missing_values_after)


import matplotlib.pyplot as plt

# Plot the distribution of MEDV
plt.figure(figsize=(10, 6))
plt.hist(df['MEDV'], bins=30, edgecolor='black', alpha=0.7)
plt.title('Distribution of Housing Prices (MEDV)')
plt.xlabel('MEDV (in $1000s)')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()

from sklearn.model_selection import train_test_split

# Split the dataset into training, validation, and test sets
train_val, test = train_test_split(df, test_size=0.2, random_state=42)
train, val = train_test_split(train_val, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

# Check the shapes of the splits
print(f'Train set: {train.shape}')
print(f'Validation set: {val.shape}')
print(f'Test set: {test.shape}')

from sklearn.preprocessing import StandardScaler

# Separating features and target
X_train = train.drop('MEDV', axis=1)
y_train = train['MEDV']
X_val = val.drop('MEDV', axis=1)
y_val = val['MEDV']
X_test = test.drop('MEDV', axis=1)
y_test = test['MEDV']

# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform all sets
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Check the scaled values
print(f'Scaled training data mean: {X_train.mean(axis=0)}')
print(f'Scaled training data std: {X_train.std(axis=0)}')


# -------------MLP

layers = [X_train.shape[1], 64, 32, 1]  # Input layer, two hidden layers, output layer

# Create an instance of the MLPRegression class
mlp = MLPRegression(layers=layers, learning_rate=0.01, activation='relu', optimizer='sgd', batch_size=32, epochs=100)

# Train the model
mlp.fit(X_train, y_train)

# Evaluate on validation set
mse_val, rmse_val, r2_val, mae_val = mlp.evaluate(X_val, y_val)
print(f'Validation MSE: {mse_val:.4f}, RMSE: {rmse_val:.4f}, R²: {r2_val:.4f}')


#  ---------------wandb

import wandb 

sweep_config = {
    'method': 'random',  # 'random', 'grid', or 'bayes'
    'metric': {
        'name': 'MSE',
        'goal': 'minimize'  # We want to minimize MSE
    },
    'parameters': {
        'learning_rate': {'min': 0.0001, 'max': 0.1},
        'optimizer': {
            'values': ['sgd', 'batch', 'mini-batch']  # Testing SGD and Adam
        },
        'activation': {
            'values': ['relu', 'tanh', 'sigmoid']  # Trying different activation functions
        },
        'batch_size': {
            'values': [16, 32, 64]  # Different batch sizes
        },
        'epochs': {
            'values': [50, 100, 200]  # Number of epochs
        },
        'layers': {
            'values': [[X_train.shape[1], 64, 32, 1], [X_train.shape[1], 10, 1], [X_train.shape[1], 32, 16, 1]]  # Different network architectures
        }
    }
}

def sweep_train():
    # Initialize a new run
    wandb.init()

    # Get hyperparameters from the W&B config
    config = wandb.config

    # Create the MLP model with hyperparameters from the sweep config
    mlp = MLPRegression(
        layers=config.layers,
        learning_rate=config.learning_rate,
        activation=config.activation,
        optimizer=config.optimizer,
        batch_size=config.batch_size,
        epochs=config.epochs
    )

    # Train the model
    mlp.fit(X_train, y_train)

    # Evaluate the model on the validation set
    mse, rmse, r2 = mlp.evaluate(X_val, y_val)

    # Log metrics to W&B
    wandb.log({"MSE": mse, "RMSE": rmse, "R²": r2})

# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project="MLP-Regression-Sweep")

# Run the sweep
wandb.agent(sweep_id, function=sweep_train, count=20)  # count defines how many runs to execute



# ----------------------- Best model

layers = [X_train.shape[1], 13, 10, 1]  # Input layer, two hidden layers, output layer

mlp = MLPRegression(layers=layers, learning_rate=0.01935, activation='relu', optimizer='batch', batch_size=64, epochs=50)

mlp.fit(X_train, y_train)

mse_test, rmse_test, r2_test, mae_test = mlp.evaluate(X_test, y_test)
print(f'Validation MSE: {mse_test:.4f}, RMSE: {rmse_test:.4f}, R²: {r2_test:.4f}, MAE: {mae_test:.4f}')



# --------------------------- MSE vs BCE


data = pd.read_csv('../../data/external/diabetes.csv')

X = data.iloc[:, :-1].values  # All columns except the last
y = data.iloc[:, -1].values   # Last column


X_train, X_testval, y_train, y_testval = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_testval, y_testval, test_size=0.5, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


wandb.init(project='diabetes-classification', entity='your_entity_name')

# Train models
model_bce = MLPBinaryClassifier(input_size=X_train.shape[1], learning_rate=0.01, loss_type='bce', epochs=100)
model_bce.fit(X_train, y_train)

model_mse = MLPBinaryClassifier(input_size=X_train.shape[1], learning_rate=0.01, loss_type='mse', epochs=100)
model_mse.fit(X_train, y_train)

# Predict and evaluate
y_pred_bce = model_bce.predict(X_test)
y_pred_mse = model_mse.predict(X_test)

accuracy_bce = np.mean(y_pred_bce.flatten() == y_test)
accuracy_mse = np.mean(y_pred_mse.flatten() == y_test)

print(f'Accuracy with BCE Loss: {accuracy_bce:.2f}')
print(f'Accuracy with MSE Loss: {accuracy_mse:.2f}')

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(model_bce.losses, label='BCE Loss')
plt.title('BCE Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(model_mse.losses, label='MSE Loss', color='red')
plt.title('MSE Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()


# ---------------------------------------- AuoEncoders ------------------------------------


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Step 1: Load the Dataset
data = pd.read_csv('../../data/external/spotify.csv')

# Step 3: Select Numeric Columns (6 to 20, excluding column 8)
numeric_data = data.iloc[:, [6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]

# Step 4: Remove Invalid Values
numeric_data = numeric_data.apply(pd.to_numeric, errors='coerce').dropna()

# Step 5: Split into Input and Output
output_column = data.iloc[numeric_data.index, -1]  # Assuming the output column is now the last one
X = numeric_data  # Input features with the output column removed
y = output_column.reset_index(drop=True)  # Output labels

# Step 7: Standardize Remaining Data
scaler = MinMaxScaler()
X_standardized = scaler.fit_transform(X)

# Step 8: Split the Data
X_train, X_temp, y_train, y_temp = train_test_split(X_standardized, y, test_size=0.2, random_state=42)
X_validate, X_test, y_validate, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Output the shapes of the resulting datasets
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_validate shape:", X_validate.shape)
print("y_validate shape:", y_validate.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)



input_size = X_train.shape[1] 
hidden_layers = []  
latent_size = 11  

# Initialize the AutoEncoder
autoencoder = AutoEncoder(
    input_size=input_size,
    hidden_layers=hidden_layers,
    latent_size=latent_size,
    activation='sigmoid',
    optimizer='mini_batch',
    learning_rate=0.01,
    batch_size=32,
    epochs=10
)

# Train the AutoEncoder
autoencoder.fit(X_train)
latent_train = autoencoder.get_latent(X_train)

latent_test = autoencoder.get_latent(X_test)




# -------------------------------------- Implement KNN



from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()
# X_train = scaler.fit_transform(latent_train)
# X_test = scaler.fit_transform(latent_test)

# Apply KNN on the reduced (latent) data
knn = KNeighborsClassifier(n_neighbors=20)  # Choose the number of neighbors
knn.fit(latent_train, y_train)

# Predict on the test set
y_pred = knn.predict(latent_test)

# Calculate metrics
f1 = f1_score(y_test, y_pred, average='macro')
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')

print(f'F1 Score: {f1}')
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')




# ----------------------------------- Implement MLP Classifier

class MLP:
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
        self.label_map = {}
        self.reverse_label_map = {}

    def _initialize_parameters(self, input_size, output_size):
        layer_sizes = [input_size] + self.hidden_layers + [output_size]
        for i in range(1, len(layer_sizes)):
            self.weights.append(np.random.randn(layer_sizes[i-1], layer_sizes[i]) * np.sqrt(2. / layer_sizes[i-1]))
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
        
        unique_labels = np.unique(y)
        n_classes = len(unique_labels)
        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
        self.reverse_label_map = {idx: label for label, idx in self.label_map.items()}
        
        y_adjusted = np.array([self.label_map[label] for label in y])
        
        self._initialize_parameters(n_features, n_classes)
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            if self.optimizer == 'sgd':
                indices = np.random.permutation(n_samples)
                X_shuffled = X[indices]
                y_shuffled = y_adjusted[indices]
                for i in range(0, n_samples, self.batch_size):
                    batch_X = X_shuffled[i:i+self.batch_size]
                    batch_y = y_shuffled[i:i+self.batch_size]
                    dW, db = self._backpropagation(batch_X, np.eye(n_classes)[batch_y])
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
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")
            
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
        return np.array([self.reverse_label_map[idx] for idx in predicted_indices])

    def accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)

    def _compute_cost(self, X, y):
        activations = self._forward_propagation(X)
        m = X.shape[0]
        epsilon = 1e-15
        cost = -np.sum(y * np.log(activations[-1] + epsilon) + (1 - y) * np.log(1 - activations[-1] + epsilon)) / m
        return cost

    def gradient_check(self, X, y, epsilon=1e-7):
        y_adjusted = np.array([self.label_map[label] for label in y])
        y_encoded = np.eye(len(self.label_map))[y_adjusted]
        
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


# Create an instance of the MLP class with specified parameters.
mlp = MLP(hidden_layers=[64, 32],
            learning_rate=0.001,
            activation='relu',
            optimizer='mini_batch',
            batch_size=32,
            epochs=100)

# Fit the model on training data.
mlp.fit(latent_train, y_train)

# Predict on the test set
y_pred_mlp = mlp.predict(latent_test)

# Evaluate the MLP classifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

f1_mlp = f1_score(y_test.values, y_pred_mlp, average='macro')
accuracy_mlp = accuracy_score(y_test.values, y_pred_mlp)
precision_mlp = precision_score(y_test.values, y_pred_mlp, average='macro')
recall_mlp = recall_score(y_test.values, y_pred_mlp, average='macro')

print(f'MLP F1 Score: {f1_mlp}')
print(f'MLP Accuracy: {accuracy_mlp}')
print(f'MLP Precision: {precision_mlp}')
print(f'MLP Recall: {recall_mlp}')

# Perform gradient checking on a sample of validation data.
# is_gradient_correct = mlp.gradient_check(X_val[:10], y_val[:10])
# print(f"Gradient check passed: {is_gradient_correct}")






