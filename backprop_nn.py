import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.metrics import mean_squared_error

# Activation Functions
def sigmoid(z):
	"""Sigmoid activation function."""
	return 1 / (1 + np.exp(-z))

def sigmoid_deriv(z):
	"""Derivative of the sigmoid function."""
	return sigmoid(z) * (1 - sigmoid(z))

def relu(z):
	"""
	ReLU activation function.
	When z > 0, ReLU returns z (slope 1). When z <= 0, ReLU returns 0 (slope 0).
	"""
	return np.maximum(0, z)

def relu_deriv(z):
	"""
	Derivative of the ReLU function.
	When z > 0, the derivative is 1. When z <= 0, the derivative is 0.
	"""
	return np.where(z > 0, 1, 0)

# Loss Function
def mse(y_true, y_pred):
	"""Mean Squared Error (MSE) loss."""
	return np.mean((y_true - y_pred) ** 2)

def mse_deriv(y_true, y_pred):
	"""Derivative of MSE loss wrt predictions."""
	return -2 * (y_true - y_pred) / y_true.size

# Create fake dataset for linear regression
def create_dataset():
	# Seed rng for reproducibility
	np.random.seed(0)
	X = np.random.rand(1, 100)  # 100 examples, 1 feature
	noise = np.random.normal(0, 0.2, size=X.shape[1])  # Match noise to the number of samples
	m, b = 3, 2  # Slope and intercept
	Y = (m * X) + b + noise  # Linear data with noise
	# Plug into pandas
	data = pd.DataFrame({'x': X.flatten(), 'y': Y.flatten()})
	return data

# Prepare plot
def init_plot(data):
	fig, ax = plt.subplots()

	# Plot the inputs (x) and observations (y)
	ax.scatter(data['x'], data['y'], color='blue', label='Observations')

	# Create a dynamic line for the NN, initially empty
	line, = ax.plot([], [], color='red', linestyle='--', label='NN Prediction')

	# Label the axes and add a title
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_title('Backpropagation from Scratch')

	# Add a legend to distinguish between lines and points
	ax.legend()

	return fig, ax, line

# Animation function using beta history
def animate(frame, line, X, nn):
	# Get the coefficients (beta) for the current frame
	weights = nn.weight_history[frame]
	nn.set_weights(weights[0], weights[1], weights[2], weights[3])

	# Calculate predictions for y using the trained network
	y_pred = nn.predict(X)

	# Update the data for the dynamic line in the plot
	line.set_data(X.flatten(), y_pred.flatten())

	return line,

class NN:
	def __init__(self, input_size, hidden_size, output_size, learning_rate=0.5):
		"""
		Initialize the NN.
		- input_size: # of input features.
		- hidden_size: # of neurons in the hidden layer.
		- output_size: # of output neurons.
		- learning_rate: Learning rate for gradient descent.
		"""
		# Initialize weights and biases randomly for the hidden and output layers
		self.w1 = np.random.randn(hidden_size, input_size)  # Weights for hidden layer
		self.b1 = np.zeros((hidden_size, 1))  # Biases for hidden layer
		self.w2 = np.random.randn(output_size, hidden_size)  # Weights for output layer
		self.b2 = np.zeros((output_size, 1))  # Biases for output layer
		self.learning_rate = learning_rate  # Learning rate for updates

		self.weight_history = []  # Store weights for animation

	# For animation
	def set_weights(self, w1, b1, w2, b2):
		self.w1 = w1
		self.b1 = b1
		self.w2 = w2
		self.b2 = b2

	def forward(self, X):
		"""
		Perform forward propagation.
		- X: Input data (shape: [input_size, num_samples]).
		Returns: Predicted output.
		"""
		# Compute the hidden layer activations
		self.z1 = np.dot(self.w1, X) + self.b1  # Linear transformation: w1 * X + b1
		self.a1 = relu(self.z1)  # Apply ReLU activation function
		
		# Compute the output layer activations
		self.z2 = np.dot(self.w2, self.a1) + self.b2  # Linear transformation: w2 * a1 + b2
		self.a2 = self.z2  # No activation (linear output for regression)

		return self.a2

	def backward(self, X, Y, output):
		"""
		Perform backward propagation to compute gradients and update weights.
		- X: Input data.
		- Y: True labels.
		- output: Predicted output from forward propagation.
		"""
		m = X.shape[1]  # num of examples

		# See README.md for variable derivations.

		# Compute gradients for the output layer
		dL_dyhat = mse_deriv(Y, output)  # Derivative of loss w.r.t. y_hat (z2)
		dL_dz2 = dL_dyhat
		
		dL_dw2 = (1 / m) * np.dot(dL_dz2, self.a1.T)  # Gradient of w2
		dL_db2 = (1 / m) * np.sum(dL_dz2, axis=1, keepdims=True)  # Gradient of b2

		# Compute gradients for the hidden layer
		dL_da1 = np.dot(self.w2.T, dL_dz2)  # Backpropagate error to hidden layer
		dL_dz1 = dL_da1 * relu_deriv(self.z1)  # Apply derivative of ReLU
		dL_dw1 = (1 / m) * np.dot(dL_dz1, X.T)  # Gradient of w1
		dL_db1 = (1 / m) * np.sum(dL_dz1, axis=1, keepdims=True)  # Gradient of b1

		# Update weights and biases using gradient descent
		self.w2 -= self.learning_rate * dL_dw2
		self.b2 -= self.learning_rate * dL_db2
		self.w1 -= self.learning_rate * dL_dw1
		self.b1 -= self.learning_rate * dL_db1

	def train(self, X, Y, epochs):
		"""
		Train the neural network using forward and backward propagation.
		- X: Input data.
		- Y: True labels.
		- epochs: # of training iterations.
		"""
		for epoch in range(epochs):
			# Perform forward propagation to compute predictions
			output = self.forward(X)

			# Compute the loss
			loss = mse(Y, output)

			# Perform backward propagation to update weights
			self.backward(X, Y, output)

			# Print the loss every 100 epochs
			if epoch % 100 == 0:
				print(f"Epoch {epoch}: Loss = {loss:.4f}")

			self.weight_history.append((self.w1.copy(), self.b1.copy(), self.w2.copy(), self.b2.copy()))

	def predict(self, X):
		"""
		Make predictions using the trained neural network.
		- X: Input data.
		Returns: Predicted outputs.
		"""
		return self.forward(X)


if __name__ == "__main__":
	# Create dataset
	data = create_dataset()
	print("Dataset Preview:\n", data.head())  # Print the first few rows of the dataset

	# Reshape X and Y to have proper dimensions
	X = data['x'].values.reshape(1, -1)  # Shape: (1, 100)
	Y = data['y'].values.reshape(1, -1)  # Shape: (1, 100)

	# Initialize and train the neural network
	nn = NN(input_size=1, hidden_size=10, output_size=1)
	nn.train(X, Y, epochs=1000)

	# Generate predictions for the data
	predicted_Y = nn.predict(X)

	# Downsample frames for the animation
	sampled_frames = np.linspace(0, len(nn.weight_history) - 1, 100, dtype=int)

	# Create the static analytical plot with an empty container for the dynamic regression plot.
	fig, ax, line = init_plot(data)

	# Animate
	anim = FuncAnimation(fig, animate, frames=sampled_frames, 
						 fargs=(line, X, nn), interval=50, blit=True)

	# Save as gif
	anim.save("backprop_nn.gif", writer=PillowWriter(fps=20))

	# Plot
	plt.show()

