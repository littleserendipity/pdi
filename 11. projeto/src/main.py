import misc.constant as const
import misc.data as data
import argparse

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset", help="pass a dataset name to use", type=str)
	parser.add_argument("--augmentation", help="how many images to augmentation", type=int)
	parser.add_argument("--train", help="train the NN", action="store_true")
	parser.add_argument("--test", help="test the NN", action="store_true", default=True)
	parser.add_argument("--gpu", help="select the GPU mode", action="store_true")
	args = parser.parse_args()

	if (args.dataset):
		const.setup(args.dataset, args.gpu)

		if (args.augmentation):
			data.augmentation(args.augmentation)

		elif (args.train):
			print("args.train", args.train)

		elif (args.test):
			print("args.test", args.test)

	else:
		print("Pass a valid dataset name")

if __name__ == '__main__':
	main()