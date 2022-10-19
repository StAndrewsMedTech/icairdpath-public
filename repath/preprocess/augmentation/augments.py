import torchvision.transforms.functional as TF
from typing import List
import random

class Rotate(object):
    """ Create a rotate class """
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img):
        image = TF.rotate(img, angle=self.angle)
        return image

class FlipRotate(object):
    """ Create a flip and rotate class """
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img):
        """ Flips and then Rotates the image.
        
        Args:
            img (Image): The input image 

        Returns:
            image (Image): flipped and rotated imagg.

        """
        image = TF.hflip(img)
        image = TF.rotate(image, angle=self.angle)
        return image


class RandomRotateFromList(object):
    def __init__(self, angles: List[float]) -> None:
        super().__init__()
        self.angles = angles

    def __call__(self, img):
        # select angle
        angle = random.choice(self.angles)
        #print("angle:", angle)

        # apply angle
        image = TF.rotate(img, angle=angle)
        return image
    
    
class RandomCropSpecifyOffset(object):
    def __init__(self, size_diff: int) -> None:
        super().__init__()
        self.size_diff = size_diff
        
    def __call__(self, img):
        #select offset
        offset = random.choice(list(range(0, self.size_diff)))
        #print("offset:", offset)
        imsize = img.size
        width = imsize[0] - self.size_diff
        height = imsize[1] - self.size_diff
        image = TF.crop(img, top=offset, left=offset, height=height, width=width)

        return image
