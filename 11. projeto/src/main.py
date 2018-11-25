from util import path, data
import control.constant as const
import control.nn as nn
import argparse

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset", help="pass a dataset name to use", type=str)
	parser.add_argument("--arch", help="pass a neural network architecture", type=str, default="unet")
	parser.add_argument("--augmentation", help="how many images to augmentation", type=int)
	parser.add_argument("--train", help="train the NN", action="store_true")
	parser.add_argument("--test", help="test the NN", action="store_true", default=True)
	parser.add_argument("--gpu", help="select the GPU mode", action="store_true")
	args = parser.parse_args()

	dn_dataset = path.data(args.dataset, mkdir=False)

	if (args.dataset and path.exist(dn_dataset)):
		const.setup(args.dataset, args.arch, args.gpu)

		if (args.augmentation):
			data.augmentation(args.augmentation)

		elif (args.train):
			nn.train()

		elif (args.test):
			nn.test()
	else:
		print("\n>> Dataset not found (%s)" % dn_dataset)

if __name__ == '__main__':
	main()