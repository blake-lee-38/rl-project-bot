import os
import zipfile
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict

# ==============================
# 1. Load and Preprocess Images
# ==============================
def load_images(data_dir, image_size=(64, 64), max_per_class=100, num_classes=None):
    X, y = [], []
    label_map = {}
    label_counter = 0

    for label_name in sorted(os.listdir(data_dir)):
        if num_classes is not None and label_counter >= num_classes:
            break
        folder = os.path.join(data_dir, label_name)
        if not os.path.isdir(folder):
            continue
        files = os.listdir(folder)
        if label_name not in label_map:
            label_map[label_name] = label_counter
            label_counter += 1
        for file in files:
            try:
                img_path = os.path.join(folder, file)
                img = Image.open(img_path).convert('RGB')  # convert to RGB
                img = img.resize(image_size)
                img_array = np.asarray(img, dtype=np.float32) / 255.0
                # Add original image
                X.append(img_array.flatten())
                y.append(label_map[label_name])
            except:
                continue
    return np.array(X), np.array(y), label_map

# ==============================
# 3. Split Data Manually
# ==============================
def split_dataset(X, y, train_frac=0.8, val_frac=0.0):
    indices = list(range(len(X)))
    random.shuffle(indices)
    n_train = int(train_frac * len(X))
    n_val = int(val_frac * len(X))

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return X[train_idx], y[train_idx], X[val_idx], y[val_idx], X[test_idx], y[test_idx]

# ==============================
# 4. Weighted k-NN From Scratch
# ==============================
class WeightedKNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        return np.array([self._predict_one(x) for x in X])

    def _predict_one(self, x):
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        k_idxs = np.argsort(distances)[:self.k]
        class_weights = defaultdict(float)

        for idx in k_idxs:
            label = self.y_train[idx]
            dist = distances[idx]
            weight = 1 / (dist + 1e-5)  # avoid division by zero
            class_weights[label] += weight

        return max(class_weights.items(), key=lambda item: item[1])[0]

    def score(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)

# ==============================
# 5. Plot Learning Curve
# ==============================
def plot_learning_curve(knn, X_train, y_train, X_val, y_val, steps=10):
    sizes = np.linspace(10, len(X_train), steps, dtype=int)
    train_scores, val_scores = [], []

    for size in sizes:
        knn.fit(X_train[:size], y_train[:size])
        train_acc = knn.score(X_train[:size], y_train[:size])
        val_acc = knn.score(X_val, y_val)
        train_scores.append(train_acc)
        val_scores.append(val_acc)

    plt.plot(sizes, train_scores, label='Training Accuracy')
    plt.plot(sizes, val_scores, label='Validation Accuracy')
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.title("Weighted k-NN Learning Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

# ==============================
# 6. Main
# ==============================
def main():
    class_amounts = [2, 3, 5, 10, 15, 30, 50]
    k_values = [1, 3, 5, 7, 9, 12, 15, 20]
    for num_classes in class_amounts:
        print(f"\nTesting with {num_classes} classes...")
        X, y, label_map = load_images("c:/Users/Blake/Desktop/VS-Code-Projects/RL_Project/SL_Project/animals/animals", image_size=(16, 16), max_per_class=100, num_classes=num_classes)

        # Split the original data
        X_train, y_train, _, _, X_test, y_test = split_dataset(X, y, train_frac=0.8, val_frac=0.0)

        best_k = None
        best_accuracy = 0

        # Train and test to find the best k
        for k in k_values:
            knn = WeightedKNN(k=k)
            knn.fit(X_train, y_train)
            test_acc = knn.score(X_test, y_test)
            print(f"Test Accuracy for k={k}: {test_acc * 100:.2f}%")

            if test_acc > best_accuracy:
                best_accuracy = test_acc
                best_k = k

        print(f"Best k for {num_classes} classes: {best_k} with Test Accuracy: {best_accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
