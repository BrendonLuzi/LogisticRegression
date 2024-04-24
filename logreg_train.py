import numpy as np
import pandas as pd
from logistic_regression import LogisticRegression

def main():
	# Load the dataset
	traning_data = pd.read_csv('datasets/dataset_train.csv')

	# Extract the input features and target variable
	y = traning_data['Hogwarts House'].values
	
	# Skip the first 6 columns
	data = traning_data.iloc[:, 6:]
	for column in data.columns:
		data[column] = data[column].fillna(data[column].mean())
	x = data.values

	# Train the model
	model = LogisticRegression()
	model.classes = np.unique(y)
	model.fit(x, y)
	
	# Load the test dataset
	test_data = pd.read_csv('datasets/dataset_test.csv')

	# Extract the input features get only lines with house Gryffindor
	data = test_data.iloc[:, 6:]
	x_test = data.fillna(data.mean()).values

	# Predict the target variable
	model.predict(x_test)

if __name__ == '__main__':
	main()