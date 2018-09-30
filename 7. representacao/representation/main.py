import Image as im
import Segmentation as sg
import Representation as rp

def main():
    s = sg.Thresholding()
    r = rp.Representation()

    ### questão 1
    img1a = s.otsu(im.Image("Image_(1).bmp"))
    
    img, chain, chain_norm = r.chain(img1a, directions=8)
    print("chain: " + str(len(chain)) + "\n", chain, "\n")
    print("chain_norm: " + str(len(chain_norm)) + "\n", chain_norm)
    img.save(extension="8_directions_(0,-1)")


if __name__ == '__main__':
    main()