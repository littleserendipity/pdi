import Image as im
import Compress as cm

def main():
    
    ### questão 1
    img1 = cm.Huffman(im.Image("Image_(1).tif"))
    img1.compress()
    img1.decompress()


if __name__ == '__main__':
    main()