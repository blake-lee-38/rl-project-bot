import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# --- Image Loading ---
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
        files = os.listdir(folder)[:max_per_class]
        if label_name not in label_map:
            label_map[label_name] = label_counter
            label_counter += 1
        for file in files:
            try:
                img_path = os.path.join(folder, file)
                img = Image.open(img_path).convert('RGB')
                img = img.resize(image_size)
                img_array = np.asarray(img, dtype=np.float32) / 255.0
                X.append(img_array.flatten())
                y.append(label_map[label_name])
            except:
                continue
    return np.array(X), np.array(y), label_map

# --- Utility Functions ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

def one_hot(y, num_classes):
    encoded = np.zeros((len(y), num_classes))
    for idx, label in enumerate(y):
        encoded[idx, label] = 1
    return encoded

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def dropout(x, drop_prob):
    keep_prob = 1 - drop_prob
    mask = np.random.rand(*x.shape) < keep_prob
    return (x * mask) / keep_prob

def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# --- Neural Network Class ---
class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.lr = learning_rate
        # Xavier initialization
        self.W1 = np.random.randn(hidden_size, input_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(output_size, hidden_size) * np.sqrt(2. / hidden_size)
        self.b2 = np.zeros((output_size, 1))

    def forward(self, x):
        self.z1 = self.W1 @ x + self.b1
        self.a1 = relu(self.z1)
        self.z2 = self.W2 @ self.a1 + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, x, y_true):
        y_pred = self.a2
        delta2 = (y_pred - y_true) * sigmoid_derivative(self.z2)
        delta1 = (self.W2.T @ delta2) * relu_derivative(self.z1)

        self.W2 -= self.lr * delta2 @ self.a1.T
        self.b2 -= self.lr * delta2
        self.W1 -= self.lr * delta1 @ x.T
        self.b1 -= self.lr * delta1

    def train(self, X, Y, X_val, Y_val, epochs=100, batch_size=32):
        for epoch in range(epochs):
            total_loss = 0
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            for start_idx in range(0, X.shape[0], batch_size):
                end_idx = min(start_idx + batch_size, X.shape[0])
                batch_indices = indices[start_idx:end_idx]
                X_batch = X[batch_indices]
                Y_batch = Y[batch_indices]
                for i in range(X_batch.shape[0]):
                    x = X_batch[i].reshape(-1, 1)
                    y = Y_batch[i].reshape(-1, 1)
                    y_pred = self.forward(x)
                    self.backward(x, y)
                    total_loss += np.sum((y - y_pred)**2)
            train_loss = total_loss / X.shape[0]
            train_accuracy = np.mean(np.argmax(self.predict(X), axis=1) == np.argmax(Y, axis=1))
            val_loss = np.mean((Y_val - self.predict(X_val))**2)
            print(f"Epoch {epoch+1}, Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, Validation Loss: {val_loss:.4f}")

    def predict(self, X):
        predictions = []
        for i in range(X.shape[0]):
            x = X[i].reshape(-1, 1)
            y_pred = self.forward(x)
            predictions.append(y_pred.flatten())
        return np.array(predictions)

def split_dataset(X, y, train_frac=0.8):
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    n_train = int(train_frac * len(X))

    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]

# --- Main ---
if __name__ == "__main__":
    data_dir = "c:/Users/Blake/Desktop/VS-Code-Projects/RL_Project/SL_Project/animals/animals"
    class_amounts = [10]  # Focus on 10 classes for overfitting analysis
    image_size = (32, 32)

    for num_classes in class_amounts:
        print(f"\nTesting with {num_classes} classes...")
        X, y, label_map = load_images(data_dir, image_size=image_size, num_classes=num_classes)
        num_classes = len(label_map)
        Y = one_hot(y, num_classes)

        # Split the data into training, validation, and test sets
        X_train, y_train, X_test, y_test = split_dataset(X, y, train_frac=0.7)
        X_train, y_train, X_val, y_val = split_dataset(X_train, y_train, train_frac=0.8)
        Y_train = one_hot(y_train, num_classes)
        Y_val = one_hot(y_val, num_classes)

        # Initialize the neural network
        nn = SimpleNeuralNetwork(input_size=X_train.shape[1], hidden_size=64, output_size=num_classes, learning_rate=0.01)

        # Track accuracies
        train_accuracies = []
        test_accuracies = []

        # Train the model
        for epoch in range(100):
            nn.train(X_train, Y_train, X_val, Y_val, epochs=1)
            train_accuracy = np.mean(np.argmax(nn.predict(X_train), axis=1) == np.argmax(Y_train, axis=1))
            test_accuracy = np.mean(np.argmax(nn.predict(X_test), axis=1) == y_test)
            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)

        # Apply moving average
        smoothed_train_accuracies = moving_average(train_accuracies)
        smoothed_test_accuracies = moving_average(test_accuracies)

        # Plot smoothed accuracy curves
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(smoothed_train_accuracies) + 1), smoothed_train_accuracies, label='Training Accuracy')
        plt.plot(range(1, len(smoothed_test_accuracies) + 1), smoothed_test_accuracies, label='Testing Accuracy')
        plt.title('Training vs Testing Accuracy for 10 Classes')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Final test accuracy
        print(f"Final test accuracy for {num_classes} classes: {test_accuracy:.2f}")
