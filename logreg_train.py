import pickle
import numpy as np
import pandas as pd
from logistic_regression import LogisticRegression


def main():
	# Load the dataset
	training_data = pd.read_csv('datasets/dataset_train.csv')

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
	# Extract only the meaningful numerical features and the target values
	target = training_data['Hogwarts House']
	training_data = training_data[features]
	
	# Fill missing values with the mean and convert to a numpy array
	training_data = training_data.fillna(training_data.mean()).values

	# Train the model
	model = LogisticRegression()
	model.fit(training_data, target)
	
	# Save the model
	with open('model.pkl', 'wb') as file:
		pickle.dump(model, file)

if __name__ == '__main__':
	main()