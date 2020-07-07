#Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
#Function for counting area of a figure
def count_area(pth):
    # load the image
    image = Image.open(pth)
    # convert image to numpy array
    data = np.asarray(image)
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
    return area_of_figure
def delete_bg(pth):
    #Parameters
    BLUR = 21
    CANNY_THRESH_1 = 10
    CANNY_THRESH_2 = 200
    MASK_DILATE_ITER = 10
    MASK_ERODE_ITER = 10
    MASK_COLOR = (0.0,0.0,1.0) # In BGR format

    #Read image
    img = cv2.imread('data/grey.jpg')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #Edge detection
    edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)

    #Find contours in edges, sort by area
    contour_info = []
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        contour_info.append((
            c,
            cv2.isContourConvex(c),
            cv2.contourArea(c),
        ))
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
    max_contour = contour_info[0]

    #Create empty mask, draw filled polygon on it corresponding to largest contour
    #Mask is black, polygon is white
    mask = np.zeros(edges.shape)
    cv2.fillConvexPoly(mask, max_contour[0], (255))

    #Smooth mask, then blur it
    mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
    mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
    mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)

    mask_stack = np.dstack([mask]*3)    # Create 3-channel alpha mask

    #Blend masked img into MASK_COLOR background
    mask_stack  = mask_stack.astype('float32') / 255.0          # Use float matrices, 
    img         = img.astype('float32') / 255.0                 #  for easy blending

    masked = (mask_stack * img) + ((1-mask_stack) * MASK_COLOR) # Blend
    masked = (masked * 255).astype('uint8')                     # Convert back to 8-bit 
    #Split image into channels
    c_red, c_green, c_blue = cv2.split(img)
    #Merge with mask got on one of a previous steps
    img_a = cv2.merge((c_red, c_green, c_blue, mask.astype('float32') / 255.0))
    #Save to disk
    cv2.imwrite('./ready_images/deleted_bg.png', img_a*255)

def calculate_area():
    print('Enter the path to your image')
    pth = input()
    delete_bg(pth)
    area = count_area('./ready_images/deleted_bg.png')
    print(f"The area of your figure is {area} cm2")
calculate_area()