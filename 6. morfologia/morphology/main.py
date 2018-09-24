import numpy as np
import Image as im
import Segmentation as seg
import Morphology as mp

def main():
    ### questão 1
    s = seg.Thresholding()
    m = mp.Morphology()

    # img1a = s.otsu(im.Image("Image_(1a).png"))
    # img1b = s.otsu(im.Image("Image_(1b).png"))

    # imgOR = m.logicalOperator(img1a, img1b, "OR")
    # imgOR.save()

    # imgAND = m.logicalOperator(img1a, img1b, "AND")
    # imgAND.save()

    # imgXOR = m.logicalOperator(img1a, img1b, "XOR")
    # imgXOR.save()

    # imgNAND = m.logicalOperator(img1a, img1b, "NAND")
    # imgNAND.save()

    ### questão 2
    # img2a = s.otsu(im.Image("Image_(2a).jpg"))
    # d_img2a = m.erode(m.erode(m.dilate(img2a)))
    # d_img2a.save()

    ### questão 3
    img3a = s.otsu(im.Image("Image_(3a).jpg"))
    # img3a = m.erode(img3a)

    d_img3a = m.erode(img3a)
    d_img3a.setImg((np.subtract(d_img3a.arr, img3a.arr)))

    d_img3a = m.erode(d_img3a, kernel=np.array([[0,1,0],[1,1,1],[0,1,0]]))


    img = m.logicalOperator(img3a, d_img3a, "OR")
    img3a.show()
    d_img3a.show()
    img.show()


if __name__ == '__main__':
    main()