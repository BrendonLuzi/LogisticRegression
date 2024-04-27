import matplotlib.pyplot as plt
import pandas as pd

def main():
	# Load the dataset
	data = pd.read_csv('datasets/dataset_train.csv')

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
	
	# Normalize the data
	data[features] = (data[features] - data[features].min()) / (data[features].max() - data[features].min())

	# Get the unique house names and define their respective colors
	houses = data['Hogwarts House'].unique()
	colors = {'Ravenclaw': 'blue', 'Slytherin': 'green', 'Gryffindor': 'red', 'Hufflepuff': 'orange'}
	# Get the color for each data point based on the house
	scatter_colors = data['Hogwarts House'].apply(lambda x: colors[x])

	# Prepare the plot and subplots
	fig, ax = plt.subplots(13, 13, figsize=(10, 10))
	# Adjust spacing and margins
	plt.subplots_adjust(hspace=0.2, wspace=0.2)
	plt.subplots_adjust(left=0.07, right=0.97, top=0.93, bottom=0.03)

	# Loop through every slot of the matrix
	for row, row_feature in enumerate(features):
		for column, col_feature in enumerate(features):
			# Plot the graphs
			if row == column:
				# If on the diagonal, plot the histogram to display the distribution of the data
				for house in houses:
					ax[row, column].hist(data[data['Hogwarts House'] == house][row_feature], alpha=0.5, label=house, color=colors[house])
			else:
				# If off the diagonal, plot a scatter plot to display the relationship between two features
				ax[row, column].scatter(data[col_feature], data[row_feature], alpha=0.3, c = scatter_colors, s = 1)
			# Remove the ticks for better readability
			ax[row, column].set_xticks([])
			ax[row, column].set_yticks([])
			# Set the labels for each row and column, adding line breaks for better readability
			if row == 0:
				ax[row, column].set_title(col_feature.replace(' ', '\n'), fontsize=7)
			if column == 0:
				ax[row, column].set_ylabel(row_feature.replace(' ', '\n'), fontsize=7)

	# Display the plot
	plt.show()

if __name__ == '__main__':
	main()
