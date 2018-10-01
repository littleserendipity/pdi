import Image as im
import Segmentation as sg
import Representation as rp

def main():
    s = sg.Thresholding()
    r = rp.Representation()

    # ### questão 1
    img1 = s.otsu(im.Image("Image_(1).bmp"))
    
    result = r.chain(img1, directions=8)
    img, chain, chain_norm = result[0], result[1], result[2]

    extension = "8_directions_(0,-1)"
    r.saveChain(img.name, extension+"_chain", chain)
    r.saveChain(img.name, extension+"_chain_norm", chain_norm)
    img.save(extension=extension)

    ### questão 2
    img2 = s.otsu(im.Image("Image_(2).jpg"))
    result = r.chain(img2, directions=4, norm=False)
    r.mpp(result[0], side=15)

if __name__ == '__main__':
    main()