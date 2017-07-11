import PIL.Image as Image
import numpy as np

def preprocess(image_arr):
    """"preprocess the image according to the original atari paper (silver et al) note: do benchmarking of this vs numpy
    preprocessing! note: at the moment i only use one image not multiple images per state"""
    img = Image.fromarray(image_arr, mode="RGB")
    img = img.convert("L")
    img = img.resize((84, 110))
    img = img.crop((0, 13, 84, 97))
    img_arr = np.asarray(img.getdata(), dtype=np.uint8).reshape((84, 84))
    return img_arr