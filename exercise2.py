# Exercise 2: Train the perceptron and plot learning progress
# This script trains a perceptron on the prepared dataset X and y
# and plots the number of misclassifications per epoch.

import matplotlib.pyplot as plt
from perceptron_class import Perceptron
from exercise1 import X, y

# Create Perceptron instance and train
perceptron = Perceptron(input_size=2, lr=0.01, epochs=20)
perceptron.fit(X, y)

# Print number of epochs
print(f"Training complete. Number of epochs: {perceptron.epochs}")

# Plot learning progress
plt.plot(range(1, perceptron.epochs + 1), perceptron.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.title('Perceptron Learning Progress')
plt.show()
