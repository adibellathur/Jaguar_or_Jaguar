from fastai.vision import *

def create_dataset(path, folder, filename):
	dest = path/folder
	dest.mkdir(parents=True, exist_ok=True)
	download_images(path/filename, dest, max_pics=100, max_workers=0)
	return

def verify_dataset(path, classes):
	for c in classes:
		print(c)
		verify_images(path/c, delete=True, max_size=100)

def main():
	path = Path('./data')
	# create_dataset(path,'leopard','leopard.csv')
	create_dataset(path,'car','car.csv')
	verify_dataset(path, ['jaguar', 'car'])


if __name__ == '__main__':
	main()
