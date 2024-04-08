import sys
import os
import keyboard
import numpy as np
import pandas as pd

class Printer:
	def __init__(self, dataset):
		self.dataset = dataset
		self.width = os.get_terminal_size().columns
		self.features = [feature for feature in dataset.columns if dataset[feature].dtype in ['int64', 'float64'] and feature != 'Index']
		self.ppterminal = (self.width - 15) // 18
		self.pages = len(self.features) // self.ppterminal - (len(self.features) % self.ppterminal == 0)
		self.page = 0

		self.count = [len([x for x in dataset[feature] if not np.isnan(x)]) for feature in self.features]
		self.mean = [np.nanmean(dataset[feature]) for feature in self.features]
		self.std = [np.nanstd(dataset[feature]) for feature in self.features]
		self.min = [np.nanmin(dataset[feature]) for feature in self.features]
		self.max = [np.nanmax(dataset[feature]) for feature in self.features]
		self.q1 = [np.nanpercentile(dataset[feature], 25) for feature in self.features]
		self.q2 = [np.nanpercentile(dataset[feature], 50) for feature in self.features]
		self.q3 = [np.nanpercentile(dataset[feature], 75) for feature in self.features]

		keyboard.on_press_key('right', lambda e:self.change_page(1))
		keyboard.on_press_key('left', lambda e:self.change_page(-1))

	def change_page(self, delta):
		if self.page + delta < 0 or self.page + delta > self.pages:
			return
		self.page += delta
		self.print_all()

	def print_feature(self) -> None:
		print(f'{'':<15}', end='')
		for i in range(self.page * self.ppterminal, (self.page + 1) * self.ppterminal):
			if i >= len(self.features):
				break
			if len(self.features[i]) > 15:
				print(f'{self.features[i][:12]:<12}...', end='   ')
			else:
				print(f'{self.features[i][:15]:<15}', end='   ')
		print()
	
	def print_data(self, list: list) -> None:
		for i in range(self.page * self.ppterminal, (self.page + 1) * self.ppterminal):
			if i >= len(list):
				break
			print(f'{str(list[i])[:10]:<18}', end='')
		print()

	def print_all(self) -> None:
		os.system('cls')
		self.print_feature()
		print(f'{"Count":<15}', end='')
		self.print_data(self.count)
		print(f'{"Mean":<15}', end='')
		self.print_data(self.mean)
		print(f'{"Std":<15}', end='')
		self.print_data(self.std)
		print(f'{"Min":<15}', end='')
		self.print_data(self.min)
		print(f'{"25%":<15}', end='')
		self.print_data(self.q1)
		print(f'{"50%":<15}', end='')
		self.print_data(self.q2)
		print(f'{"75%":<15}', end='')
		self.print_data(self.q3)
		print(f'{"Max":<15}', end='')
		self.print_data(self.max)
		print(f'Page: {self.page + 1}/{self.pages + 1}')

		
def main():
	# dataset = pd.read_csv(sys.argv[1])
	dataset = pd.read_csv('datasets/dataset_train.csv')
	printer = Printer(dataset)
	printer.print_all()
	keyboard.wait('q', suppress=True)
	
if __name__ == '__main__':
	main()