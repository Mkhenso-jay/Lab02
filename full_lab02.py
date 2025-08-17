import numpy as np
import matplotlib.pyplot as plt

# ---------- Perceptron Class ----------
class Perceptron:
    def __init__(self, input_size, lr=0.01, epochs=20):
        self.lr = lr
        self.epochs = epochs
        self.weights = np.zeros(input_size + 1)  # weights + bias

    def activation(self, x):
        return 1 if x >= 0 else -1  # 1 = Popular, -1 = Not Popular

    def predict(self, x):
        z = np.dot(x, self.weights[1:]) + self.weights[0]
        return self.activation(z)

    def fit(self, X, y):
        self.errors_ = []
        for epoch in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y):
                update = self.lr * (target - self.predict(xi))
                self.weights[1:] += update * xi
                self.weights[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

# ---------- Training Dataset ----------
X = np.array([
    [1, 5], [2, 8], [1, 10], [2, 12], [3, 15],
    [1, 7], [2, 9], [3, 11], [4, 12], [1, 6],
    [2, 10], [3, 14], [1, 8], [2, 13], [3, 15],
    [4, 18], [1, 7], [2, 9], [3, 12], [4, 14],
    [1, 6], [2, 8], [3, 10], [4, 11], [2, 7],

    # Popular (0–100 scale for reviews)
    [6, 12], [7, 29], [8, 64], [6, 15], [7, 70],
    [8, 159], [9, 150], [10, 17], [7, 100], [6, 5],
    [8, 91], [9, 47], [10, 66], [7, 157], [6, 41],
    [8, 25], [9, 169], [10, 95], [7, 21], [6, 88],
    [8, 190], [9, 37], [10, 115], [7, 120], [6, 58]
])

y = np.array([
    # -1 = Not Popular
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,
    
    # 1 = Popular
    1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1
])

# Shuffle dataset
indices = np.arange(len(X))
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

# ---------- Create and Train Perceptron ----------
perceptron = Perceptron(input_size=2, lr=0.01, epochs=20)
perceptron.fit(X, y)

# ---------- Print Epoch Errors ----------
print("Epoch Errors:")
for epoch, errors in enumerate(perceptron.errors_, start=1):
    print(f"Epoch {epoch}: {errors} misclassifications")

# ---------- Learning Progress ----------
plt.plot(range(1, perceptron.epochs + 1), perceptron.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.title('Perceptron Learning Progress')
plt.grid(True)
plt.show()

# ---------- Decision Boundary Plot ----------
def plot_decision_boundary(X, y, model, new_book=None):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 10, X[:, 1].max() + 10
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 1))

    Z = np.array([model.predict(np.array([i, j])) for i, j in zip(xx.ravel(), yy.ravel())])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8,6))
    plt.contourf(xx, yy, Z, alpha=0.3, levels=np.linspace(-1, 1, 3), colors=['#FF9999','#99FF99'])
    
    for label, color in zip([-1,1], ['red','green']):
        plt.scatter(X[y==label, 0], X[y==label, 1], c=color, label=f'{"Not Popular" if label==-1 else "Popular"}', s=80, edgecolor='k')
    
    if new_book is not None:
        pred = model.predict(new_book)
        color = 'green' if pred == 1 else 'red'
        plt.scatter(new_book[0], new_book[1], c=color, s=250, marker='*', edgecolor='k', label='New Book')
    
    # Plot decision boundary line
    if model.weights[2] != 0:
        x_vals = np.array([x_min, x_max])
        y_vals = -(model.weights[1] * x_vals + model.weights[0]) / model.weights[2]
        plt.plot(x_vals, y_vals, 'b--', label='Decision Boundary')

    plt.text(x_min+0.5, y_max-15, f"Weights: {model.weights[1]:.2f}, {model.weights[2]:.2f}\nBias: {model.weights[0]:.2f}", fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
    plt.xlabel('Author Popularity')
    plt.ylabel('Number of Reviews')
    plt.title('Book Popularity Decision Boundary')
    plt.legend()
    plt.grid(True)
    plt.show()

# ---------- User Input Prediction ----------
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

# Call the user input function
user_input_prediction(perceptron)
