from control import nn, generator, constant
from util import path, data
import argparse

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--tolabel", help="Preprocess images to create labels (out/tolabel)", action="store_true", default=False)
	parser.add_argument("--dataset", help="Dataset name", type=str)
	parser.add_argument("--arch", help="Neural Network architecture", type=str, default="unet")
	parser.add_argument("--augmentation", help="Dataset augmentation (pass quantity)", type=int)
	parser.add_argument("--train", help="Train", action="store_true")
	parser.add_argument("--validation", help="Enable validation step", action="store_true", default=False)
	parser.add_argument("--test", help="Predict", action="store_true", default=True)
	parser.add_argument("--gpu", help="Enable GPU mode", action="store_true")
	args = parser.parse_args()

	if (args.tolabel):
		dn_tolabel = path.out(constant.dn_TOLABEL, mkdir=False)

		if path.exist(dn_tolabel):
			generator.tolabel()
		else:
			print("\n>> Folder not found (%s)\n" % dn_tolabel)

	elif (args.dataset is not None):
		dn_dataset = path.data(args.dataset, mkdir=False)

		if (path.exist(dn_dataset)):
			constant.setup(args)

			if (args.augmentation):
				generator.augmentation(args.augmentation)

			elif (args.train):
				nn.train()
				
			elif (args.test):
				nn.test()
	else:
		print("\n>> Dataset not found\n")

if __name__ == '__main__':
	main()