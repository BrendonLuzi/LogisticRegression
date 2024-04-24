import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LogisticRegression:

	# Default values
	LEARNING_RATE = 0.01
	EPOCHS = 100000
	PRECISION = 0.000001

	def __init__(self, learning_rate=LEARNING_RATE, epochs=EPOCHS, precision=PRECISION):
		self.learn_rate = learning_rate
		self.epochs = epochs
		self.precision = precision
		self.weights = {}
		self.classes = None
		self.train_min = None
		self.train_max = None
		self.probabilities = None
		self.predictions = None
		self.loss_history = {}

	def __sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	def __loss(self, y, y_pred):
		return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

	def __normalize(self, x, min = None, max = None):
		# If min and max are not provided, retrieve them from the data
		if min is None or max is None:
			self.train_min = np.min(x, axis=0)
			self.train_max = np.max(x, axis=0)

		# Normalize the data
		return (x - self.train_min) / (self.train_max - self.train_min)
	
	def __gradient_descent(self, x, y):
		# Get the length of the features
		n = x.shape[0]
		
		# Loop through each class to perform one-vs-all gradient descent
		for c in self.classes:
			# Get the binary target values for the current class
			y_bin = (y == c).astype(int)

			# Initialize the weights randomly and the loss history array
			self.weights[c] = np.random.rand(x.shape[1])
			self.loss_history[c] = []

			# Perform gradient descent until convergence or max epochs
			for _ in range(self.epochs):
				# Predict the probabilities of being in the current class
				y_pred = self.__sigmoid(np.dot(x, self.weights[c]))
				# Calculate the gradient
				error = y_bin - y_pred
				gradient = -np.dot(error, x) / n
				# Update the weights
				delta_weight = self.learn_rate * gradient
				self.weights[c] -= delta_weight
				# Calculate the loss and store it
				self.loss_history[c].append(self.__loss(y_bin, y_pred))

				# Check for convergence through the change in loss
				if len(self.loss_history[c]) > 2 and abs(self.loss_history[c][-2] - self.loss_history[c][-1]) < self.precision:
					print(f'Converged for class {c} at epoch {_}')
					break
			
			# If the model did not converge, print a warning
			if len(self.loss_history[c]) == self.epochs:
				print(f'Model did not converge for class {c}')
	
	def fit(self, x, y):
		# Normalize the data and add a column of ones for the bias coefficient
		x = self.__normalize(x)
		x = np.insert(x, 0, 1, axis=1)		

		# Get the unique classes
		self.classes = np.unique(y)

		# Train the model
		self.__gradient_descent(x, y)

	def save_probabilities(self):
		# Create a log file to display the probabilities
		with open('extra.log', 'w') as file:
			# Loop through all the samples
			for i in range(len(self.probabilities[0])):
				# Print the sample index and the predicted class
				print(f'Sample {i}\tis in class {self.predictions[i]}.', end='\t', file = file)
				# Print the probability of being in each class as ordered rounded percentages
				print('Probabilities:', end=' ', file = file)
				# Sort the probabilities in descending order
				probs = np.argsort(self.probabilities, axis=0)[:, i][::-1]
				# Print the probabilities in descending order
				for j in probs:
					print(f'{self.classes[j]}: {round(self.probabilities[j][i] * 100, 2)}%', end=',\t', file = file)
				print(file = file)

	def predict(self, x):
		# Normalize the data according to the training min and max
		x = self.__normalize(x, self.train_min, self.train_max)
		# Add a column of ones for the bias coefficient
		x = np.insert(x, 0, 1, axis=1)
  
		# Find probability of each class
		self.probabilities = [self.__sigmoid(np.dot(x, self.weights[c])) for c in self.classes]

		self.predictions = []
		# For each sample, predict its class and store it
		for i in range(len(self.probabilities[0])):
			prediction = self.classes[np.argmax([self.probabilities[j][i] for j in range(len(self.probabilities))])]
			self.predictions.append(prediction)

		return self.predictions
	
	def visualize_gradient_descent(self):
		# Create a subplot for each class and put space between them
		fig, ax = plt.subplots(len(self.classes), 1, figsize=(10, 8))
		plt.subplots_adjust(hspace=0.5)

		# Set the labels for each subplot
		for i, c in enumerate(self.classes):
			ax[i].set_title(c)
			ax[i].set_ylabel('Loss')
		ax[-1].set_xlabel('Epochs')

		# Turn on interactive mode for live plotting
		plt.ion()

		# Find extremes needed for plotting and choose colors
		colors = ['red', 'orange', 'blue', 'green']
		max_epochs = max([len(self.loss_history[c]) for c in self.classes])
		max_loss = max([max(self.loss_history[c]) for c in self.classes])
		min_loss = min([min(self.loss_history[c]) for c in self.classes])

		# Create a line for each class
		lines = {}
		for i, c in enumerate(self.classes):
			# Initialize the line with no data
			lines[i], = ax[i].plot([], [], label=c, color=colors[i])
			# Set the x and y limits for each subplot
			ax[i].set_xlim(0, max_epochs)
			# ax[i].set_ylim(0, max_loss)

			# Set the y-axis to logarithmic scale
			ax[i].set_ylim(min_loss, max_loss)
			ax[i].set_yscale('log')

		# Update the plot every 1% of epochs
		for i in range(0, max_epochs, max_epochs // 100):
			# Update the data of each line if it exists
			for j, c in enumerate(self.classes):
				if i < len(self.loss_history[c]):
					lines[j].set_xdata(range(i+1))
					lines[j].set_ydata(self.loss_history[c][:i+1])
			# Update the plot and pause for a short time
			plt.draw()
			plt.pause(0.00001)

		# Turn off interactive mode
		plt.ioff()
		# Display the plot and wait for the user to close it
		plt.show()