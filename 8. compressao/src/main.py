import Image as im
import Compress as cm

def main():
    
    huffman = cm.Huffman()
    lzw = cm.LZW()

    ### questão 2
    # img1 = im.Image("Image_(1).tif")
    # compress = huffman.compress(img1)
    # huffman.decompress(compress)

    # img2 = im.Image("Image_(2).tif")
    # compress = huffman.compress(img2)
    # huffman.decompress(compress)

    # img3 = im.Image("Image_(3).tif")
    # compress = huffman.compress(img3)
    # huffman.decompress(compress)

    ### questão 2
    img1 = im.Image("Image_(1).tif")
    compress = lzw.compress(img1)
    # lzw.decompress(compress)

if __name__ == '__main__':
    main()