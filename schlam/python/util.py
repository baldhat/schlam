import cv2
import torch
import numpy as np

def build_pyramid(image, num_levels, scale = 0.5):
    '''
    :param image:
    :param num_levels: Number of images in the pyramid
    :param scale: Number by which width and height of the original image gets multiplied
    :return: A list of resized images starting with the largest image
    '''
    assert(num_levels >= 1)
    assert(scale < 1)
    batched = False
    grayscale = False
    if len(image.shape) == 4:
        if image.shape[0] > 1:
            raise RuntimeError("Can only handle single image batches")
        batched = True
        image = image[0]
    if image.shape[0] == 1:
        image = image[0]
        grayscale = True
    pyramid = [image.cpu().numpy().astype(np.uint8)]
    for i in range(1, num_levels):
        new_w = int(round(pyramid[-1].shape[-1] * scale))
        new_h = int(round(pyramid[-1].shape[-2] * scale))
        pyramid.append(cv2.resize(pyramid[-1], (new_w, new_h), interpolation = cv2.INTER_LINEAR))
    pyramid = [torch.tensor(image_).to(image.device) for image_ in pyramid]
    if grayscale:
        pyramid = [image_.unsqueeze(0) for image_ in pyramid]
    if batched:
        pyramid = [image_.unsqueeze(0) for image_ in pyramid]
    return pyramid