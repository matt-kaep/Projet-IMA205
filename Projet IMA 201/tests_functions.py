import tempfile
import IPython
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage import filters
import cv2
import time
import random
from tqdm import tqdm
import preprocessing_functions as pre
import SRM_segmentation_functions as SRM
import pandas as pd
import skimage.morphology as morpho
import parametres as dullrazor
import valentin as val

def test_database(database,FilePath,database_correction,canal):
    liste_images=[]
    for FileName in database:
        im = io.imread(FilePath+FileName)
        image_preprocessed = pre.preprocessing(im, (150, 100))
        regions,regions_updated = SRM.SRM_segmentation_3canaux(image_preprocessed,50,canal)
        image_region = SRM.creer_image_regions_3canaux(regions,regions_updated)
        liste_images.append(image_region)
    # Calculate the number of rows and columns
    num_images = len(liste_images)
    grid_size = math.ceil(math.sqrt(num_images))

    fig, ax = plt.subplots(grid_size, grid_size, figsize=(20, 20))

    for i in range(grid_size):
        for j in range(grid_size):
            index = i * grid_size + j
            if index < num_images:
                ax[i, j].imshow(liste_images[index])
                ax[i, j].set_title(database[index])
            else:
                ax[i, j].axis('off')  # Hide empty subplots
    plt.show()
    
    return liste_images,df_dice_coefficient
    

def dice_coefficient(image1, image2):
    image2=(pre.resize_image(image2,(150,100))>0).astype(int)
    intersection = np.logical_and(image1, image2)
    return 2 * intersection.sum() / (image1.sum() + image2.sum())


def test_database(database_à_tester, database_correction, FilePath,canal):
    df_dice_coeffcient = pd.DataFrame(columns=['Image', 'Dice'])
    liste_images_dilatees = []
    start_time = time.time()
    for i, image_name in enumerate(database_à_tester):
        im = io.imread(FilePath + image_name)
        im_co = io.imread(FilePath + database_correction[database_à_tester.index(image_name)])
        image_preprocessed_val = (val.pre_process_temp(FilePath + image_name)*255).astype(np.uint8)
        image_preprocessed_nico=dullrazor.dullrazor(image_preprocessed_val)
        image_preprocessed=pre.preprocessing(image_preprocessed_nico,(150,100))
        regions, regions_updated = SRM.SRM_segmentation_3canaux(image_preprocessed, 60, canal)
        regions = pre.remplacement_fond(regions, regions_updated)
        image_regions_binaire_final = SRM.binary_image(regions, regions_updated,canal)
        image_dilatee = SRM.dilatation(morpho.disk(2), image_regions_binaire_final)
        liste_images_dilatees.append(image_dilatee)
        dice = dice_coefficient(image_dilatee, im_co)
        df_dice_coeffcient = df_dice_coeffcient.append({'Image': image_name, 'Dice': dice}, ignore_index=True)
        plt.figure(i)
        plt.subplot(1, 2, 1)
        plt.imshow(image_dilatee)
        plt.title('Image obtenue')
        plt.subplot(1, 2, 2)
        plt.imshow(im_co)
        plt.title('Image correction')
        plt.show()
        print("Le coeffcient dice est :", dice)
        elapsed_time = time.time() - start_time
        remaining_images = len(database_à_tester) - (i + 1)
        estimated_remaining_time = elapsed_time / (i + 1) * remaining_images
        print(f"Estimated remaining time: {estimated_remaining_time} seconds")
    return liste_images_dilatees, df_dice_coeffcient

######ESSAI PARALLELISATION########
from multiprocessing import Pool, cpu_count

def process_image(args):
    print("debut du traitement de l'image", args[0])
    image_name, database_correction, FilePath,database_à_tester,canal = args
    im = io.imread(FilePath + image_name)
    im_co = io.imread(FilePath + database_correction[database_à_tester.index(image_name)])
    image_preprocessed = pre.preprocessing(im, (150, 100))
    regions, regions_updated = SRM.SRM_segmentation_3canaux(image_preprocessed, 30, canal)
    regions = pre.remplacement_fond(regions, regions_updated)
    print(image_name)
    image_regions_binaire_final = SRM.binary_image(regions, regions_updated,canal)
    image_dilatee = SRM.dilatation(morpho.disk(2), image_regions_binaire_final)
    dice = dice_coefficient(image_dilatee, im_co)
    return image_name, image_dilatee, dice

def test_database_multiprocessing(database_à_tester, database_correction, FilePath,canal):
    df_dice_coeffcient = pd.DataFrame(columns=['Image', 'Dice'])
    liste_images_dilatees = []

    with Pool(cpu_count()) as p:
        results = p.map(process_image, [(image_name, database_correction, FilePath,database_à_tester,canal) for image_name in database_à_tester])

    for image_name, image_dilatee, dice in results:
        liste_images_dilatees.append(image_dilatee)
        df_dice_coeffcient = df_dice_coeffcient.append({'Image': image_name, 'Dice': dice}, ignore_index=True)

    return liste_images_dilatees, df_dice_coeffcient

