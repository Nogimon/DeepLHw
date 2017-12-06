import imageio
from glob import glob


if __name__ == "__main__":
    filename = glob("./savedexample2/*[0-9].jpg")
    images = []
    for file in filename:
        images.append(imageio.imread(file))
    imageio.mimsave("./movie2mask.gif", images)