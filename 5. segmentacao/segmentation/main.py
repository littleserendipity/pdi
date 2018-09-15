import Image as im
import Edge as ed

def main():
    e = ed.Edge()

    ### 1ª questão
    img = im.Image("Image_(1).jpg", noise=1)
    e_img = e.laplaceofGaussian(img)
    e_img.save(extension="edge")

    img = im.Image("Image_(1a).jpg", noise=2)
    e_img = e.laplaceofGaussian(img)
    e_img.save(extension="edge")

    img = im.Image("Image_(2a).jpg", noise=3)
    e_img = e.laplaceofGaussian(img)
    e_img.save(extension="edge")

    ### 2ª questão
    # img = im.Image("Image_(3a).jpg", noise=0)
    # e_img = e.laplaceofGaussian(img)
    # e_img.save(extension="edge")

    # img = im.Image("Image_(3b).jpg", noise=21)
    # e_img = e.laplaceofGaussian(img)
    # e_img.save(extension="edge")

if __name__ == '__main__':
    main()