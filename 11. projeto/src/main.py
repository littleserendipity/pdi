import argparse
import control.constant as const
import control.nn as nn
import util.data as data

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset", help="pass a dataset name to use", type=str)
	parser.add_argument("--arch", help="pass a neural network architecture", type=str, default="unet")
	parser.add_argument("--augmentation", help="how many images to augmentation", type=int)
	parser.add_argument("--train", help="train the NN", action="store_true")
	parser.add_argument("--test", help="test the NN", action="store_true", default=True)
	parser.add_argument("--gpu", help="select the GPU mode", action="store_true")
	args = parser.parse_args()

	if (args.dataset):
		const.setup(args.dataset, args.arch, args.gpu)

		if (args.augmentation):
			data.augmentation(args.augmentation)

		elif (args.train):
			nn.train()

		elif (args.test):
			nn.test()
	else:
		print("Pass a valid dataset name")

if __name__ == '__main__':
	main()