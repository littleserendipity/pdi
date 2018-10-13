import Image as im

def main():

    img = im.Image("Image_(1a)", type="jpg")
    img.light(10)
    img.show()
    # img.save(extension="brighten_hsi")

    # img = im.Image("Image_(1b)", type="jpg")
    # img.light(0.4)
    # img.show()
    # img.save(extension="darken_hsi")

if __name__ == '__main__':
    main()