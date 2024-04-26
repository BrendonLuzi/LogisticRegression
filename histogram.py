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

	# Get the unique house names and their respective colors
	houses = data['Hogwarts House'].unique()
	colors = {'Ravenclaw': 'blue', 'Slytherin': 'green', 'Gryffindor': 'red', 'Hufflepuff': 'orange'}

	# Extract data for each house
	for house in houses:
		house_data = [data[data['Hogwarts House'] == house] for house in houses]
	
	# Prepare the plot and subplots to display the histograms
	fig, ax = plt.subplots(3, 5, figsize=(15, 9))
	# Adjust spacing and margins
	plt.subplots_adjust(hspace=0.5, wspace=0.5)
	plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

	# Plot the histograms for each feature
	# Loop through every subplot
	for i in range(len(ax.flatten())):
		# Calculate the row and column index of the subplot
		row, column = i // 5, i % 5
		# If the index is within the number of features, plot the histogram
		if i <len(features):
			feature = features[i]
			# Plot the histogram for each house
			for h_data, house in zip(house_data, houses):
				# Plot the histogram, set the transparency to 0.5 for better visibility
				ax[row, column].hist(h_data[feature], alpha=0.5, label=house, color=colors[house])
			ax[row, column].set_title(feature)
		else:
			# Remove the axis if there is no data to display
			ax[row, column].axis('off')

	# Add a legend to the plot and display it
	fig.legend(houses, loc='lower right', bbox_to_anchor=(0.9, 0.1))
	plt.show()

if __name__ == '__main__':
	main()
