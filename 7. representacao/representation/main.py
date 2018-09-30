import Image as im
import Segmentation as sg
import Representation as rp

def main():
    s = sg.Thresholding()
    r = rp.Representation()

    ### questão 1
    img1a = s.otsu(im.Image("Image_(1).bmp"))
    
    img, chain, chain_norm = r.chain(img1a, directions=8)
    extension = "8_directions_(0,-1)"
    r.saveChain(img.name, extension+"_chain", chain)
    r.saveChain(img.name, extension+"_chain_norm", chain_norm)
    img.save(extension=extension)

if __name__ == '__main__':
    main()