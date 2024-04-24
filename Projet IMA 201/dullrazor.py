import cv2
import matplotlib.pyplot as plt

#IMAGE ACQUISITION

def dullrazor(image):
    #Gray scale
    grayScale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #Black hat filter
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)) 
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    #Gaussian filter
    bhg = cv2.GaussianBlur(blackhat, (3, 3), cv2.BORDER_DEFAULT)
    #Binary thresholding (MASK)
    ret, mask = cv2.threshold(bhg, 10, 255, cv2.THRESH_BINARY)
    #Replace pixels of the mask
    dst = cv2.inpaint(image, mask, 6, cv2.INPAINT_TELEA)  
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB) 
    return dst

