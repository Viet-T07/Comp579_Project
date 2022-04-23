import torch as T
from PIL import Image
from scipy.interpolate import interp1d
import numpy as np
import scipy.misc as smp



actor = T.load("model/actor.pth")
actor.eval


m = interp1d([-1,1],[0,256])
list_of_pixel = [] 
for param in actor.parameters():
    for elt in param:        
        if len(param.shape) > 1:
            for value in elt:
                val = m(value.item())
                color = (val, 128, 256-val)
                list_of_pixel.append(color)
        else:
            val = m(elt.item())
            color = (val, 256-val, 128)
            list_of_pixel.append(color)



# Create a 1024x1024x3 array of 8 bit unsigned integers
data = np.zeros((256,128,3), dtype=np.uint8 )

for x in range(data.shape[0]):
    for y in range(data.shape[1]):
        index = x*data.shape[1] + y
        if index >= len(list_of_pixel):
            break
        data[x,y] = list_of_pixel[index]

image = Image.fromarray(data)  # Create a PIL image
image.show()                      # View in default viewer

image.save("weight_image.png")