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
    # img3a = s.otsu(im.Image("Image_(3a).jpg"))

    # d_img3a = m.floodFill(img3a, (0,0), 1)
    # d_img3a.save(extension="floodFill")

    # d_img3a.setImg(np.logical_not(d_img3a.arr))
    # d_img3a.save(extension="floodFill_inverse")

    # img = m.logicalOperator(img3a, d_img3a, "OR")
    # img.save(extension="floodFill")




    ### TESTE
    # img3a = s.otsu(im.Image("teste.jpg"))
    # # img3a = m.erode(img3a)
    
    # c_img3a = im.Image(np.logical_not(img3a.arr), name=img3a.name)
    # # c_img3a.show()

    # # img3a.setImg(np.logical_not(img3a.arr))
    # # img3a.show()

    # d_img3a = m.dilate(img3a)
    # d_img3a = m.dilate(d_img3a)
    # d_img3a = m.dilate(d_img3a)
    # d_img3a = m.dilate(d_img3a)
    # d_img3a = m.dilate(d_img3a)
    # d_img3a = m.dilate(d_img3a)

    # img3a.show()
    # c_img3a.show()
    # d_img3a.show()

    # img = m.logicalOperator(img3a, d_img3a, "OR")

    # img3a = s.otsu(im.Image("Image_(3a).jpg"))
    # img3a = m.erode(img3a)
    # img3a.setImg(np.logical_not(img3a.arr))


    # i_img3a = im.Image(np.logical_not(img3a.arr), name=img3a.name)
    # i_img3a.show()

    # d_img3a = m.dilate(i_img3a)
    # d_img3a.setImg((np.subtract(d_img3a.arr, img3a.arr)))

    # d_img3a.setImg((np.multiply(d_img3a.arr, img3a.arr)))

    # d_img3a = m.erode(d_img3a, kernel=np.array([
    #     [0,1,0],
    #     [1,1,1],
    #     [0,1,0],
    # ]))

    # d_img3a.show()

    # img = m.logicalOperator(img3a, d_img3a, "OR")
    # img.show()

    

    ### questão 4
    kernel = np.array([
        [1,0,0,0,0],
        [0,1,0,1,0],
        [0,0,1,0,0],
        [0,0,1,0,0],
        [0,0,1,0,0],
    ])

    img4a = s.otsu(im.Image("Image_(4a).jpg"))
    # img4a.setImg(np.logical_not(img4a.arr))

    img4a = m.erode(img4a, kernel=kernel)

    # d_img4a = m.floodFill(img4a, (0,0), 1)
    # d_img4a.setImg(np.logical_not(d_img4a.arr))
    # img = m.logicalOperator(img4a, d_img4a, "OR")


    d_img4a = m.erode(img4a, kernel=kernel)


    img4a.show()
    d_img4a.show()


if __name__ == '__main__':
    main()