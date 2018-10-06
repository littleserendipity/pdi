import Image as im
import Compress as cm

def main():
    
    huffman = cm.Huffman()

    ### questão 1
    img1 = im.Image("Image_(3).tif")
    compress = huffman.compress(img1)
    huffman.decompress(compress)


if __name__ == '__main__':
    main()