import cv2
import numpy as np


IMAGE_SHAPE = (70,70)
def preprocessing_fun(img):
    img = img.astype(np.uint8)
    
    border = 170
        
    ret,thresh = cv2.threshold(img,border,255,cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    erode = cv2.erode(thresh, kernel, iterations = 2)
    img = cv2.bitwise_or(img, erode)
    

    gray = 255*(img < border).astype(np.uint8)
    coords = cv2.findNonZero(gray)
    x, y, w, h = cv2.boundingRect(coords)
    rect = img[y:y+h, x:x+w]
    
    img = cv2.copyMakeBorder(rect,1,1,1,1,cv2.BORDER_CONSTANT,value=255)
    img = cv2.resize(img, IMAGE_SHAPE, interpolation = cv2.INTER_NEAREST)
    
    img = img.astype(int)
    img = np.where(img < 230, img - 120, img)
    img = np.where(img < 0, 0, img)
    img = img.astype(np.uint8)
    img = cv2.bitwise_not(img)
    img = img.reshape(*IMAGE_SHAPE,1)
    
    return img