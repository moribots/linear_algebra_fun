import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.metrics import mean_squared_error

# Simulated dataset creation
def create_dataset():
    np.random.seed(0)  # For reproducibility
    x = np.linspace(0, 10, 20)  # Generate 20 points between 0 and 10
    m, b = 3.0, 5.0
    y = m * x + b
    data = pd.DataFrame({'x': x, 'y': y})
    return data

# Compute coefficients using the Normal Equation
def normal_equation(X, y):
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    return beta

# Gradient Descent Function with Visualization
def gradient_descent_animated(X, y, learning_rate, iterations):
    m, n = X.shape
    beta = np.zeros(n)  # Initialize coefficients
    beta_history = [beta.copy()]  # Store coefficients at each iteration
    for i in range(iterations):
        gradient = (1 / m) * X.T @ (X @ beta - y)
        beta -= learning_rate * gradient
        beta_history.append(beta.copy())
        if i % 100 == 0:  # Print progress every 100 iterations
            print(f"Iteration {i}: Coefficients: {beta}")
    return beta, beta_history

# Plot initialization
def init_plot(data, beta_normal):
    fig, ax = plt.subplots()
    ax.scatter(data['x'], data['y'], color='blue', label='Data points')
    
    # Plot the normal equation solution (static)
    X = np.c_[np.ones(len(data['x'])), data['x']]
    y_pred_normal = X @ beta_normal
    ax.plot(data['x'], y_pred_normal, color='green', label='Normal Equation')

    line, = ax.plot([], [], color='red', linestyle='--', label='Gradient Descent')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Linear Regression with Gradient Descent Animation')
    ax.legend()
    return fig, ax, line

# Animation function
def animate(frame, line, X, beta_history):
    beta = beta_history[frame]
    y_pred = X @ beta
    line.set_data(X[:, 1], y_pred)
    return line,

# Main function
def main():
    # Create dataset
    data = create_dataset()
    print("Dataset Preview:\n", data.head())

    # Prepare data for regression
    X = np.c_[np.ones(len(data['x'])), data['x']]  # Add intercept term
    y = data['y']

    # Normal Equation
    beta_normal = normal_equation(X, y)
    print(f"\nCoefficients (Normal Equation): {beta_normal}")

    # Gradient Descent
    beta_gd, beta_history = gradient_descent_animated(X, y, learning_rate=0.02, iterations=1000)
    print(f"\nCoefficients (Gradient Descent): {beta_gd}")

    # Initialize plot
    fig, ax, line = init_plot(data, beta_normal)

    # Create the animation
    anim = FuncAnimation(fig, animate, frames=len(beta_history), 
                         fargs=(line, X, beta_history), interval=10, blit=True)

    plt.show()

if __name__ == "__main__":
    main()
