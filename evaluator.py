from sklearn.metrics import accuracy_score
import pandas as pd

def main():
	# Load the csv file with the true values
	true = pd.read_csv('datasets/dataset_truth.csv')['Hogwarts House'].values
	# Load the csv file with the predicted values
	pred = pd.read_csv('houses.csv')['Hogwarts House'].values

	# Calculate the accuracy
	accuracy = accuracy_score(true, pred)
	print(f'Accuracy: {accuracy * 100}%')
	if accuracy >= 0.98:
		print('Congratulations! You have passed the test.')
	else:
		print('It\'s not good enough...')
 
if __name__ == '__main__':
	main()