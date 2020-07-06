from PIL import Image
from numpy import asarray
import pandas as pd
# load the image
image = Image.open('./ready_images/deleted_bg.png')
# convert image to numpy array
data = asarray(image)
shape = list(data.shape)
counter = 0
pixels = []
for i in data:
    for j in i:
        pixels.append(j)
for i in pixels:
    r = i[0]
    g = i[1]
    b = i[2]
    a = i[3]
    if a >= 0:
        counter+=1
ruler_length = 10 #cm
width = shape[1]
px = ruler_length/width
px_area = px**2
area_of_figure = px_area*counter