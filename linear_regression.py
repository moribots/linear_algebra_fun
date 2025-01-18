import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.metrics import mean_squared_error

# Create fake dataset for linear regression
def create_dataset():
    # Seed rng for reproducibility
    np.random.seed(0)
    
    # Generate points
    x = np.linspace(0, 10, 20)
    
    # Define the model: y = ax + b with added Gaussian noise
    # Slope (m) and intercept (b) of the model
    a, b = 3.0, 5.0  
    # mean 0 and std 2
    noise = np.random.normal(0, 2, size=len(x))  
    # Calculate noisy observations
    y = a * x + b + noise  

    # Plug into pandas
    data = pd.DataFrame({'x': x, 'y': y})
    return data

# Compute coefficients analytically (see README.md)
def normal_equation(X, y):
    # closed-form Normal Equation:
    # beta = (X^T X)^(-1) X^T y
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    return beta

# Perform gradient descent to learn the coefficients.
def gradient_descent_animated(X, y, learning_rate, iterations):
    # Get the number of samples (m) and features (n) from X
    m, n = X.shape

    # Initialize coefficients to zero
    beta = np.zeros(n)

    # Store beta's evolution for visualization
    beta_history = [beta.copy()]

    # Gradient descent
    for i in range(iterations):
        # Compute the gradient of the cost function (Mean Squared Error)
        gradient = (1 / m) * X.T @ (X @ beta - y)

        # Update the coefficients using the gradient and the learning rate
        beta -= learning_rate * gradient

        # Store the update for viz
        beta_history.append(beta.copy())

        # Print progress every iterations/10 iterations
        if i % (iterations/10) == 0:
            print(f"Iteration {i}: Coefficients: {beta}")

    return beta, beta_history

# Prepare plot
def init_plot(data, beta_normal):
    fig, ax = plt.subplots()

    # Plot the inputs (x) and observations (y)
    ax.scatter(data['x'], data['y'], color='blue', label='Observations')

    # Prepare the input matrix X for line fitting
    # Add a column of ones to include the intercept term
    # If we don't do this, the line will pass through the origin
    X = np.c_[np.ones(len(data['x'])), data['x']]

    # Compute predictions using the coefficients from the Normal Equation
    y_pred_normal = X @ beta_normal

    # Plot the analytical fit
    ax.plot(data['x'], y_pred_normal, color='green', label='Normal Equation (analytical fit)')

    # Create a dynamic line for gradient descent, initially empty
    line, = ax.plot([], [], color='red', linestyle='--', label='Gradient Descent')

    # Label the axes and add a title
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Linear Regression with Gradient Descent')

    # Add a legend to distinguish between lines and points
    ax.legend()

    return fig, ax, line

# Animation function using beta history
def animate(frame, line, X, beta_history):
    # Get the coefficients (beta) for the current frame
    beta = beta_history[frame]

    # Calculate predictions for y using the current beta values
    y_pred = X @ beta

    # Update the data for the dynamic line in the plot
    line.set_data(X[:, 1], y_pred)

    return line,

# Main function
def main():
    # Create the dataset
    data = create_dataset()
    print("Dataset Preview:\n", data.head())  # Print the first few rows of the dataset

    # Prepare data for regression
    # Add a column of ones to the feature matrix X for the intercept term
    # If we don't do this, the line will pass through the origin
    X = np.c_[np.ones(len(data['x'])), data['x']]
    
    # Extract the observations (y)
    y = data['y']

    # Solve for coefficients using the Normal Equation
    beta_normal = normal_equation(X, y)
    print(f"\nCoefficients (Normal Equation): {beta_normal}")

    # Solve for coefficients using Gradient Descent
    beta_gd, beta_history = gradient_descent_animated(X, y, learning_rate=0.02, iterations=1000)
    print(f"\nCoefficients (Gradient Descent): {beta_gd}")

    # Downsample frames for the animation
    # To avoid creating too many frames, sample 100 evenly spaced frames from the history
    sampled_frames = np.linspace(0, len(beta_history) - 1, 100, dtype=int)

    # Create the static analytical plot with an empty container for the dynamic regression plot.
    fig, ax, line = init_plot(data, beta_normal)

    # Animate
    anim = FuncAnimation(fig, animate, frames=sampled_frames, 
                         fargs=(line, X, beta_history), interval=50, blit=True)

    # Save as gif
    anim.save("gradient_descent.gif", writer=PillowWriter(fps=20))

    # Plot
    plt.show()

if __name__ == "__main__":
    main()
