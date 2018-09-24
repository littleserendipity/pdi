import numpy as np
import Image as im
import Segmentation as seg
import Morphology as mp

def main():
    ### questão 1
    s = seg.Thresholding()
    m = mp.Morphology()

    img1a = s.otsu(im.Image("Image_(1a).png"))
    img1b = s.otsu(im.Image("Image_(1b).png"))

    imgOR = m.logicalOperator(img1a, img1b, "OR")
    imgOR.save()

    imgAND = m.logicalOperator(img1a, img1b, "AND")
    imgAND.save()

    imgXOR = m.logicalOperator(img1a, img1b, "XOR")
    imgXOR.save()

    imgNAND = m.logicalOperator(img1a, img1b, "NAND")
    imgNAND.save()

    ### questão 2
    

if __name__ == '__main__':
    main()