import platform
import tempfile
import os
from scipy.optimize import minimize
from skimage.filters import threshold_otsu
from skimage.filters import try_all_threshold

import numpy as np
from matplotlib import pyplot as plt
from skimage import io as skio
from skimage import data, filters, color, morphology
from skimage.transform import rescale, resize, downscale_local_mean

def shading(im) :
    #im=skio.imread(im)
    hsv = color.rgb2hsv(im)
    
    cs = 20 #taille des bords récupérés
    
    (l, w, n) = hsv.shape
    
    c1 = hsv[:cs, :cs]
    c2 = hsv[-cs:, :cs]
    c3 = hsv[:cs, -cs:]
    c4 = hsv[-cs:, -cs:]

    s1 = c1.reshape(-1, 3)
    s2 = c2.reshape(-1, 3)
    s3 = c3.reshape(-1, 3)
    s4 = c4.reshape(-1, 3)

    k = 0

    for i in range(cs):
        for j in range(cs):
            s1[k] = (c1[i][j][2], i, j)
            s2[k] = (c2[i][j][2], l-cs+i, j)
            s3[k] = (c3[i][j][2], i, w-cs+j)
            s4[k] = (c4[i][j][2], l-cs+i, w-cs+j)
            
            k = k+1
            
    s = np.concatenate((s1, s2, s3, s4))

    p = np.zeros(6)
    e = np.zeros(((2*cs)**2))
    opt = 0

    def minim(P):
        
        def z(x,y):
            return P[0]*x**2 + P[1]*y**2 + P[2]*x*y + P[3]*x + P[4]*y + P[5]

        for i in range((2*cs)**2):
            e[i] = (s[i][0] - z(s[i][1], s[i][2]))**2
        
        opt = sum(e)
        return opt

    sol = minimize(minim, p, method='Powell')
    opP = sol.x
    #print(opP)

    for x in range(l):
        for y in range(w):
            hsv[x][y][2] /= (opP[0]*x**2 + opP[1]*y**2 + opP[2]*x*y + opP[3]*x + opP[4]*y + opP[5])

    hsv = np.clip(hsv, 0, 1)
    im2 = color.hsv2rgb(hsv)
    
    return im2

def goodskin(img):
    
    (h, w, s) = img.shape
    s = 0.02
    l = 0.25

    min_var = 0
    goodskin = []

    pas = round(w*s)
    max_iter = round(l*w/pas)

    t = []

    for i in range(max_iter):

        bu = img[i*pas:i*pas+pas, i*pas:w-i*pas]
        bu = bu.reshape(-1, 3)

        bd = img[h-(i+1)*pas:h-i*pas, i*pas:w-i*pas]
        bd = bd.reshape(-1, 3)

        bl = img[i*pas:h-i*pas, i*pas:i*pas+pas]
        bl = bl.reshape(-1, 3)

        br = img[i*pas:h-i*pas, w-(i+1)*pas:w-i*pas]
        br = br.reshape(-1, 3)

        all_b = np.concatenate((bu, bd, bl, br))
        #print(all_b.shape)

        mean = [0, 0, 0]
        std = [0, 0, 0]
        ratio = [0, 0, 0]

        for j in range(3):
            mean[j] = np.mean(all_b[:, j])
            std[j] = np.std(all_b[:, j])
            ratio[j] = std[j]/mean[j]

        varcoef = abs(np.sum(ratio))
        #print(varcoef)

        if (min_var == 0) or (varcoef < min_var) :
            min_var = varcoef
            goodskin = all_b
            best_mean = mean

    return min_var, best_mean

def corner_black(image):
    
    top_left_corner = image[20:40, 0:20]
    
    if np.median(top_left_corner[0] < 20):
        return 1
    else:
        return 0


def vignette_temp(img, s=150, v=[255, 255, 255]) :
    
    h = img.shape[0]-1
    w = img.shape[1]-1

    for i in range(round(h/2.3)+200):
        for j in range(max(round((h/2.3)-i), 200)):

            if img[i, j][2] < s:  
                img[i, j] = v
            if img[h-i, j][2] < s:  
                img[h-i, j] = v
            if img[i, w-j][2] < s:  
                img[i, w-j] = v
            if img[h-i, w-j][2] < s:  
                img[h-i, w-j] = v
    
    return img

def pre_process_temp(im):
    
    im = skio.imread(im)
    
    if corner_black(im) == 1:
        _, best_mean = goodskin(im)
        im = vignette_temp(im, v=best_mean)
        print('Vignettes noires détectées')
        
    im2 = shading(im)
    
    return im2