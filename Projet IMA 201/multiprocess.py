import time
import pandas as pd
import numpy as np
import cv2 as cv
from skimage import io
from skimage import morphology as morpho
import preprocessing_functions as pre
import dullrazor as dr
import SRM_segmentation_functions as SRM



######ESSAI PARALLELISATION########
from multiprocessing import Pool, cpu_count

def process_image(args):
    print("debut du traitement de l'image", args[0])
    image_name,canal = args
    im = io.imread(image_name)
    image_preprocessed = pre.preprocessing(im, (150, 100))
    image_dullrazored = dr.dullrazor(image_preprocessed)
    image_dullrazored_rgb = cv.cvtColor(image_dullrazored, cv.COLOR_BGR2RGB)
    regions, regions_updated = SRM.SRM_segmentation_3canaux(image_dullrazored_rgb, 30,canal)
    regions = pre.remplacement_fond(regions, regions_updated)
    print(image_name)
    image_regions_binaire_final = SRM.binary_image(regions, regions_updated,canal)
    image_dilatee = SRM.dilatation(morpho.disk(2), image_regions_binaire_final)
    return image_name, image_dilatee, image_dullrazored_rgb,image_regions_binaire_final

def test_database_multiprocessing(database_à_tester,canal):
    liste_images_finales = []
    liste_images_names = []
    liste_images_initiales = []
    liste_images_dullrazored = []
    liste_regions_binaire_final = []
    print([(image_name) for image_name in database_à_tester])
    with Pool(cpu_count()) as p:
        results = p.map(process_image, [(image_name,canal) for image_name in database_à_tester])
    for image_name, image_dilatee, image_dullrazored_rgb, image_regions_binaire_final in results:
        liste_images_initiales.append(io.imread(image_name))
        liste_images_finales.append(image_dilatee)
        liste_images_names.append(image_name)
        liste_images_dullrazored.append(image_dullrazored_rgb)
        liste_regions_binaire_final.append(image_regions_binaire_final)
    return liste_images_finales, liste_images_names, liste_images_initiales, liste_images_dullrazored, liste_regions_binaire_final