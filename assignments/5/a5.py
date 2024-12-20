
# ----------------------------- Q 2 : KDE ------------------------------

import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


# 2.1 
class KDE:
    def __init__(self, kernel='gaussian', bandwidth=1.0):
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.data = None

    def fit(self, X):
        self.data = np.array(X)

    def _kernel_function(self, distances):
        if self.kernel == 'gaussian':
            return np.exp(-0.5 * (distances / self.bandwidth) ** 2) / (self.bandwidth * np.sqrt(2 * np.pi))
        elif self.kernel == 'box':
            return np.where(np.abs(distances) <= self.bandwidth, 0.5 / self.bandwidth, 0)
        elif self.kernel == 'triangular':
            return np.maximum(1 - np.abs(distances) / self.bandwidth, 0)
        else:
            raise ValueError("Unsupported kernel type. Choose 'gaussian', 'box', or 'triangular'.")

    def predict(self, X):
        X = np.array(X)
        distances = cdist(X, self.data)
        kernel_values = self._kernel_function(distances)
        density = kernel_values.mean(axis=1)
        return density

    def visualize(self, X, grid_size=100, dim=[8, 6]):
        if X.shape[1] != 2:
            raise ValueError("Visualization is supported only for 2D data.")
        
        x_min, x_max = X[:, 0].min(), X[:, 0].max()
        y_min, y_max = X[:, 1].min(), X[:, 1].max()
        x_grid, y_grid = np.meshgrid(
            np.linspace(x_min, x_max, grid_size),
            np.linspace(y_min, y_max, grid_size)
        )
        grid_points = np.c_[x_grid.ravel(), y_grid.ravel()]
        densities = self.predict(grid_points).reshape(grid_size, grid_size)
        
        plt.figsize=(dim[0], dim[1])
        plt.scatter(X[:, 0], X[:, 1], alpha=0.5, label="Data Points")
        plt.contourf(x_grid, y_grid, densities, levels=50, cmap='coolwarm', alpha=0.7)
        plt.colorbar(label="Density")
        plt.title(f"{self.kernel.capitalize()} KDE (Bandwidth={self.bandwidth}, kernel={self.kernel})")
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.legend()
        plt.show()

# 2.2
def generate_synthetic_data():
    num_points_large = 3000
    radius_large = 2.0
    noise_large = 0.2
    angles_large = np.random.uniform(0, 2 * np.pi, num_points_large)
    radii_large = np.sqrt(np.random.uniform(0, radius_large**2, num_points_large))
    x_large = radii_large * np.cos(angles_large) + np.random.normal(0, noise_large, num_points_large)
    y_large = radii_large * np.sin(angles_large) + np.random.normal(0, noise_large, num_points_large)

    num_points_small = 500
    radius_small = 0.35
    noise_small = 0.1
    angles_small = np.random.uniform(0, 2 * np.pi, num_points_small)
    radii_small = np.sqrt(np.random.uniform(0, radius_small**2, num_points_small))
    x_small = radii_small * np.cos(angles_small) + np.random.normal(1, noise_small, num_points_small)
    y_small = radii_small * np.sin(angles_small) + np.random.normal(1, noise_small, num_points_small)

    x_combined = np.concatenate([x_large, x_small])
    y_combined = np.concatenate([y_large, y_small])

    data = np.column_stack((x_combined, y_combined))
    labels = np.array([0] * num_points_large + [1] * num_points_small)
    return data, labels

data, labels = generate_synthetic_data()

plt.figure(figsize=(8, 8))
plt.scatter(data[labels == 0][:, 0], data[labels == 0][:, 1], alpha=0.5, s=10, color="blue", label="Large Circle")
plt.scatter(data[labels == 1][:, 0], data[labels == 1][:, 1], alpha=0.7, s=10, color="blue", label="Small Circle")
plt.title("Synthetic Dataset: Two Circular Regions")
plt.xlabel("X1")
plt.ylabel("X2")
plt.axis("equal")
plt.legend()
plt.show()


# 2.3

import numpy as np
from scipy.stats import multivariate_normal

class GMM:
    def __init__(self, n_components, max_iter=100, tol=1e-3):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        self.means = None
        self.covariances = None

    def fit(self, X):
        n_samples, n_features = X.shape
        self.weights = np.full(self.n_components, 1 / self.n_components)
        self.means = X[np.random.choice(n_samples, self.n_components, replace=False)]
        self.covariances = np.array([np.cov(X, rowvar=False)] * self.n_components)

        log_likelihood_old = -np.inf
        for iteration in range(self.max_iter):
            responsibilities = self._expectation(X)
            self._maximization(X, responsibilities)
            log_likelihood = self._compute_log_likelihood(X)
            if np.abs(log_likelihood - log_likelihood_old) < self.tol:
                break
            log_likelihood_old = log_likelihood

    def _expectation(self, X):
        n_samples = X.shape[0]
        probabilities = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            probabilities[:, k] = self.weights[k] * multivariate_normal.pdf(
                X, mean=self.means[k], cov=self.covariances[k]
            )

        responsibilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        return responsibilities

    def _maximization(self, X, responsibilities):
        n_samples = X.shape[0]
        for k in range(self.n_components):
            responsibility = responsibilities[:, k]
            total_responsibility = responsibility.sum()

            self.weights[k] = total_responsibility / n_samples
            self.means[k] = (X * responsibility[:, np.newaxis]).sum(axis=0) / total_responsibility
            diff = X - self.means[k]
            self.covariances[k] = (responsibility[:, np.newaxis] * diff).T @ diff / total_responsibility

    def _compute_log_likelihood(self, X):
        log_likelihood = 0
        for k in range(self.n_components):
            log_likelihood += self.weights[k] * multivariate_normal.pdf(
                X, mean=self.means[k], cov=self.covariances[k]
            )
        return np.sum(np.log(log_likelihood))

    def predict_proba(self, X):
        probabilities = np.zeros((X.shape[0], self.n_components))

        for k in range(self.n_components):
            probabilities[:, k] = self.weights[k] * multivariate_normal.pdf(
                X, mean=self.means[k], cov=self.covariances[k]
            )

        return probabilities.sum(axis=1)
    
    def visualize_clusters(self, X):
        if X.shape[1] != 2:
            raise ValueError("Visualization is supported only for 2D data.")

        responsibilities = self._expectation(X)
        cluster_labels = np.argmax(responsibilities, axis=1)

        plt.figure(figsize=(8, 6))
        for k in range(self.n_components):
            cluster_data = X[cluster_labels == k]
            plt.scatter(cluster_data[:, 0], cluster_data[:, 1], s=10, label=f"Cluster {k + 1}")
            plt.scatter(self.means[k, 0], self.means[k, 1], c='red', s=100, marker='X', label=f"Centroid {k + 1}")

        plt.title(f"GMM Clustering with {self.n_components} Components")
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.show()

def plot_kde_comparisons(data):
    kernels = ['gaussian', 'box', 'triangular']
    bandwidths = [0.5, 0.75]
    for i, kernel in enumerate(kernels):
        for j, bandwidth in enumerate(bandwidths):
            kde = KDE(kernel=kernel, bandwidth=bandwidth)
            kde.fit(data)
            kde.visualize(data, grid_size=100, dim = [6, 4])

def plot_gmm_comparisons(data):
    components = [2, 4, 6, 8]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for i, n_components in enumerate(components):
        gmm = GMM(n_components=n_components)
        gmm.fit(data)
        responsibilities = gmm._expectation(data)
        cluster_labels = np.argmax(responsibilities, axis=1)
        for k in range(n_components):
            cluster_data = data[cluster_labels == k]
            axes[i].scatter(cluster_data[:, 0], cluster_data[:, 1], s=10, label=f"Cluster {k + 1}")
            axes[i].scatter(
                gmm.means[k, 0], gmm.means[k, 1], c='red', s=100, marker='X', label=f"Centroid {k + 1}"
            )
        axes[i].set_title(f"GMM Clustering (n_components={n_components})")
        axes[i].set_xlabel("X1")
        axes[i].set_ylabel("X2")

    plt.tight_layout()
    plt.show()

data, _ = generate_synthetic_data()

plot_kde_comparisons(data)
plot_gmm_comparisons(data)



# ---------------------------------------- Q3 : HMM ----------------------


# !pip install hmmlearn

import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from hmmlearn import hmm

#------------------ Unzipping dataset

import zipfile
import os
def unzip_file(zip_path, extract_to):
    
    if not os.path.exists(zip_path):
        print(f"Error: The file {zip_path} does not exist.")
        return

    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            print(f"File extracted to {extract_to}")
    except zipfile.BadZipFile:
        print("Error: The file is not a valid ZIP file.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Usage
zip_file_path = '../../data/external/recordings.zip'
destination_folder = '../../data/external/'
unzip_file(zip_file_path, destination_folder)

# ---------------------------




def retrieve_files(directory, initial_digit):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.startswith(str(initial_digit)):
                file_paths.append(os.path.join(root, file))
    return file_paths


def extract_mfcc(audio_path):
    audio, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=512, hop_length=256)
    return mfcc.T


def display_mfcc_spectrogram(mfcc_data, title="MFCC Spectrogram", color_map='jet'):
    plt.figure(figsize=(10, 4))
    plt.imshow(mfcc_data.T, aspect='auto', cmap=color_map, origin='lower')
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("MFCC Coefficients")
    plt.colorbar(format="%+2.0f dB")
    plt.show()


digits = [str(i) for i in range(10)]
mfcc_features_per_digit = {digit: [] for digit in digits}

for digit in digits:
    files = retrieve_files('../../data/external/recordings', digit)
    if not files:
        print(f"[Warning]: No files found for digit '{digit}' in 'recordings' directory.")
        continue

    for file in files:
        mfcc_features_per_digit[digit].append(extract_mfcc(file))

for digit, mfcc_list in mfcc_features_per_digit.items():
    if mfcc_list:
        display_mfcc_spectrogram(mfcc_list[0], title=f"MFCC for Digit {digit}")


def split_data(mfcc_data, test_size=0.2):
    train_set, test_set = {}, {}
    
    for digit, features in mfcc_data.items():
        train_set[digit], test_set[digit] = train_test_split(features, test_size=test_size, random_state=42)
    
    return train_set, test_set


train_data, test_data = split_data(mfcc_features_per_digit)

for digit in digits:
    print(f"Digit {digit}: Training samples = {len(train_data[digit])}, Testing samples = {len(test_data[digit])}")

# fit
def fit(train_data, n_components=5):
    hmm_models = {}
    
    for digit, features in train_data.items():
        concatenated_data = np.concatenate(features)
        lengths = [len(f) for f in features]
        
        model = hmm.GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=100)
        model.fit(concatenated_data, lengths)
        
        hmm_models[digit] = model
    
    return hmm_models


models = fit(train_data)

# predict
def predict(mfcc_features, hmm_models):
    best_score = float('-inf')
    predicted_label = None
    
    for digit, model in hmm_models.items():
        score = model.score(mfcc_features)
        if score > best_score:
            best_score = score
            predicted_label = digit
    
    return predicted_label

# metric
def evaluate_model(test_data, models):
    correct_predictions = 0
    total_predictions = 0
    
    for digit, features_list in test_data.items():
        for features in features_list:
            predicted_digit = predict(features, models)
            if predicted_digit == digit:
                correct_predictions += 1
            total_predictions += 1
    
    accuracy = correct_predictions / total_predictions
    return accuracy


accuracy = evaluate_model(test_data, models)
print(f"Model Recognition Accuracy: {accuracy * 100:.2f}%")


# ------------------------- Personal Recording


digits = [str(i) for i in range(10)]
mfcc_features_per_digit = {digit: [] for digit in digits}

for digit in digits:
    files = retrieve_files('../../data/external/personal_recording', digit)
    if not files:
        print(f"[Warning]: No files found for digit '{digit}' in directory.")
        continue

    for file in files:
        mfcc_features_per_digit[digit].append(extract_mfcc(file))
        
accuracy = evaluate_model(mfcc_features_per_digit, models)
print(f"Model Recognition Accuracy for Human Model: {accuracy * 100:.2f}%")
