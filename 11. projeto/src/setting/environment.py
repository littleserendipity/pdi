import setting.constant as const
import os

def setup(args):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" if args.gpu else "-1"

    const.MODEL = args.arch
    const.DATASET = args.dataset
    const.IMG_PROCESSING = args.pdi

    const.fn_CHECKPOINT = ("%s_%s_%s" % (const.MODEL, const.DATASET, const.fn_CHECKPOINT))
    const.fn_LOGGER = ("%s_%s_%s" % (const.MODEL, const.DATASET, const.fn_LOGGER))

    print("\n##################")
    print("Arch:", const.MODEL)
    print("Dataset:", const.DATASET)
    print("PDI:", const.IMG_PROCESSING)
    print("##################\n")