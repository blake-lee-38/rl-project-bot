import matplotlib.pyplot as plt
# Data from plots.py
# Format: (number of classes, testing accuracy)
data = [
    (2, 85.26),
    (3, 69.44),
    (5, 61.67),
    (10, 51.89),
    (15, 45.21),
    (30, 38.33),
    (50, 32.00)
]

# Calculate performance multiplier
performance_multipliers = [(num_classes, accuracy / (100 / num_classes)) for num_classes, accuracy in data]

# Extract data for plotting
classes = [x[0] for x in performance_multipliers]
multipliers = [x[1] for x in performance_multipliers]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(classes, multipliers, marker='o')
plt.title('Testing Accuracy as a Multiple of Random Guess')
plt.xlabel('Number of Classes')
plt.ylabel('Performance Multiplier')
plt.grid(True)
plt.xticks(classes)
plt.show()