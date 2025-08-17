# Exercise 3: Visualize decision boundary and predict new book popularity
# This script plots the perceptron's decision boundary for the dataset X and y,
# allows the user to input a new book, predicts its popularity, and shows it on the plot.

import numpy as np
import matplotlib.pyplot as plt
from perceptron_class import Perceptron
from exercise1 import X, y  # Make sure X and y are numpy arrays

# ------------------ Decision Boundary Plot ------------------
def plot_decision_boundary(X, y, model, new_book=None):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 10, X[:, 1].max() + 10
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 1))

    Z = np.array([model.predict(np.array([i, j])) for i, j in zip(xx.ravel(), yy.ravel())])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8,6))
    
    # Color regions
    plt.contourf(xx, yy, Z, alpha=0.3, levels=np.linspace(-1, 1, 3), colors=['#FF9999','#99FF99'])
    
    # Existing data points
    for label, color in zip([-1, 1], ['red', 'green']):
        plt.scatter(X[y==label, 0], X[y==label, 1], c=color, s=80, edgecolor='k', label='Popular' if label==1 else 'Not Popular')
    
    # New book
    if new_book is not None:
        pred = model.predict(new_book)
        color = 'green' if pred == 1 else 'red'
        plt.scatter(new_book[0], new_book[1], c=color, s=250, marker='*', edgecolor='k', label='New Book')

    # Decision boundary line
    if model.weights[2] != 0:  # avoid division by zero
        x_vals = np.array([x_min, x_max])
        y_vals = -(model.weights[1] * x_vals + model.weights[0]) / model.weights[2]
        plt.plot(x_vals, y_vals, 'b--', label='Decision Boundary')

    # Show weights
    plt.text(x_min+0.5, y_max-15, f"Weights: {model.weights[1]:.2f}, {model.weights[2]:.2f}\nBias: {model.weights[0]:.2f}", 
             fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    plt.xlabel('Author Popularity')
    plt.ylabel('Number of Reviews')
    plt.title('Book Popularity Decision Boundary')
    plt.legend()
    plt.grid(True)
    plt.show()

# ------------------ User Input Prediction ------------------
def user_input_prediction(model):
    print("\nEnter details of a new book to predict if it will be popular:")

    while True:
        try:
            popularity = int(input("Author Popularity (1-10): "))
            if not 1 <= popularity <= 10:
                raise ValueError
            break
        except ValueError:
            print("Please enter an integer between 1 and 10.")

    while True:
        try:
            reviews = int(input("Number of Reviews: "))
            if reviews < 0:
                raise ValueError
            break
        except ValueError:
            print("Please enter a non-negative integer.")

    new_book = np.array([popularity, reviews])
    prediction = model.predict(new_book)
    print(f"\nPrediction: {'Popular' if prediction == 1 else 'Not Popular'}")

    plot_decision_boundary(X, y, model, new_book)

# ------------------ Main Execution ------------------
if __name__ == "__main__":
    perceptron_model = Perceptron(input_size=2)
    perceptron_model.fit(X, y, epochs=10, lr=0.01)   
    user_input_prediction(perceptron_model)
