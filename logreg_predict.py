import sys
import pickle
import numpy as np
import pandas as pd
from logistic_regression import LogisticRegression

def main():
	# Load the dataset to test the model
	test_data = pd.read_csv('datasets/dataset_test.csv')

	features = ['Arithmancy', 
			 	'Astronomy', 
				'Herbology', 
				'Defense Against the Dark Arts', 
				'Divination', 
				'Muggle Studies', 
				'Ancient Runes', 
				'History of Magic', 
				'Transfiguration', 
				'Potions', 
				'Care of Magical Creatures', 
				'Charms', 
				'Flying']
	# Extract only the meaningful numerical features
	test_data = test_data[features]

	# Fill missing values with the mean and convert to a numpy array
	test_data = test_data.fillna(test_data.mean()).values

	# Load the model, and if it does not exist, quit
	try:
		with open('model.pkl', 'rb') as file:
			model = pickle.load(file)
	except FileNotFoundError:
		print('Model not found. Please train the model first.')
		return
	except:
		print('An error occurred while loading the model.')
		return
	
	# Perform predictions
	predictions = model.predict(test_data)

	# Save the predictions to a CSV file
	predictions = pd.DataFrame(predictions, columns=['Hogwarts House'])
	predictions.index.name = 'Index'
	predictions.to_csv('houses.csv')

	# If the function is called as --extra, save the probabilities of each class
	if len(sys.argv) > 1 and sys.argv[1] == '--extra':
		model.save_probabilities()

if __name__ == '__main__':
	main()
