import misc.constant as const
import argparse
import os

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset", help="pass a dataset name to use", type=str)
	parser.add_argument("--augmentation", help="process to data augmentation", action="store_true")
	parser.add_argument("--train", help="train the NN", action="store_true")
	parser.add_argument("--test", help="test the NN", action="store_true", default=True)
	parser.add_argument("--gpu", help="select the GPU mode", action="store_true")
	args = parser.parse_args()

	if (args.dataset):
		const.DATASET = args.dataset
		const.MODEL_CHECKPOINT = ("unet_%s.hdf5" % args.dataset)
		const.TXT_REPORT = ("unet_%s.txt" % args.dataset)

		os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
		os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
		os.environ["CUDA_VISIBLE_DEVICES"] = "0" if args.gpu else "-1"

		if (args.augmentation):
			print("args.augmentation", args.augmentation)

		elif (args.train):
			print("args.train", args.train)

		elif (args.test):
			print("args.test", args.test)

	else:
		print("Pass a valid dataset name")

if __name__ == '__main__':
	main()