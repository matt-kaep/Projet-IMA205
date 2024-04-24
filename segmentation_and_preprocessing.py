import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import io
import cv2 as cv
import os
import DarkArtefactRemoval as dca
import dullrazor as dr


from sklearn.cluster import KMeans
from skimage.morphology import remove_small_objects
from tqdm import tqdm



def five_segmentation(image):
    # Seuillage avec le canal bleu (RVB)
    blue_channel = image[:, :, 0]
    _, thresholded_blue = cv.threshold(blue_channel, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    thresholded_blue = cv.bitwise_not(thresholded_blue)/255

    # Seuillage avec le canal b (CIE-Lab)
    lab_image = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    b_channel = lab_image[:, :, 2]
    _, thresholded_b = cv.threshold(b_channel, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    thresholded_b = cv.bitwise_not(thresholded_b)/255

    # Regroupement des couleurs 3D avec CIE-XYZ
    xyz_image = cv.cvtColor(image, cv.COLOR_BGR2XYZ)
    xyz_features = xyz_image.reshape((-1, 3))
    x,y,z = cv.split(xyz_image)
    # Reshape x, y, z back to their original 2D shapes
    x = x.reshape(image.shape[:2])
    y = y.reshape(image.shape[:2])
    z = z.reshape(image.shape[:2])

    #threshold sur x, y, z
    _, thresholded_x = cv.threshold(x, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    thresholded_x = cv.bitwise_not(thresholded_x)/255
    _, thresholded_y = cv.threshold(y, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    thresholded_y = cv.bitwise_not(thresholded_y)/255
    _, thresholded_z = cv.threshold(z, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    thresholded_z = cv.bitwise_not(thresholded_z)/255

    return thresholded_blue, thresholded_b, thresholded_x, thresholded_y, thresholded_z


# Remove small elements of the segmentation mask
def remove_small_parts_and_fill(five_masks,min_size=1000):
    five_masks_list = list(five_masks)
    #Opening
    radius = 3
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))

    for i in range(len(five_masks_list)):
        five_masks_list[i] = five_masks_list[i].astype(np.uint8)
        #Opening
        five_masks_list[i] = cv.morphologyEx(five_masks_list[i], cv.MORPH_CLOSE, kernel)
        five_masks_list[i] = five_masks_list[i].astype(bool)
        five_masks_list[i] = remove_small_objects(five_masks_list[i], min_size=min_size)
    return five_masks_list

def union_mask(five_masks):
    mask_state = [True, True, True, True, True]

    #Rule 1 : Segmentation results with the lesion mask growing into the image border are rejected
    # Check if the lesion mask grows into the image border
    if np.any(five_masks[0][0, :]) or np.any(five_masks[0][-1, :]) or np.any(five_masks[0][:, 0]) or np.any(five_masks[0][:, -1]):
        mask_state[0] = False

    if np.any(five_masks[1][0, :]) or np.any(five_masks[1][-1, :]) or np.any(five_masks[1][:, 0]) or np.any(five_masks[1][:, -1]):
        mask_state[1] = False

    if np.any(five_masks[2][0, :]) or np.any(five_masks[2][-1, :]) or np.any(five_masks[2][:, 0]) or np.any(five_masks[2][:, -1]):
        mask_state[2] = False

    if np.any(five_masks[3][0, :]) or np.any(five_masks[3][-1, :]) or np.any(five_masks[3][:, 0]) or np.any(five_masks[3][:, -1]):
        mask_state[3] = False

    if np.any(five_masks[4][0, :]) or np.any(five_masks[4][-1, :]) or np.any(five_masks[4][:, 0]) or np.any(five_masks[4][:, -1]):
        mask_state[4] = False



    #rule 2: segmentation results without any detected region are rejected
    for i in range(0, len(five_masks)):
        if np.sum(five_masks[i]) == 0:
            mask_state[i] = False

    #rule 4: The segmentation result with the smallest mask is rejected
    smallest_mask_index = np.argmin([np.sum(mask) for mask in five_masks])
    # Reject the segmentation result with the smallest mask
    mask_state[smallest_mask_index] = False


    #rule 5: Segmentation results, whose mask areas differ too much to the other segmentation results are rejected;
    # Calculate the mask areas
    mask_areas = [np.sum(mask) for mask in five_masks]
    # Calculate the mean mask area
    mean_mask_area = np.mean(mask_areas)
    # Calculate the difference in mask areas compared to the mean
    for i in range(0, len(mask_areas)):
        if mask_areas[i] < 0.5 * mean_mask_area or mask_areas[i] > 1.5 * mean_mask_area:
            mask_state[i] = False 
    if mask_state == [False, False, False, False, False]:
        i = np.argmin([mask_areas - mean_mask_area])    
        mask_state[i] = True
        

    # building the final_mask
    for i in range(0, len(mask_state)):
        mask_true = [five_masks[i] for i in range(len(mask_state)) if mask_state[i] == True]
    united_mask = np.logical_or.reduce(mask_true)

    return united_mask


def postprocessing(image, mask_dca):
    mask_dca = cv.bitwise_not(mask_dca)
    masked_image = cv.bitwise_and(image, image, mask=mask_dca)
    return masked_image

def inpainting_dca(image):
    # Perform DCA
    dca_mask = dca.get_mask(image)
    if dca_mask is None:
        return image,None
    dca_mask_np = np.array(dca_mask, dtype=np.uint8)
    inpainted_image = cv.inpaint(image, dca_mask_np, 3, cv.INPAINT_TELEA)
    return inpainted_image,dca_mask_np


def compute_segmentation(image_path):
    image = io.imread(image_path)
    image_cleaned = dr.dullrazor(image)
    image_cleaned_rgb = cv.cvtColor(image_cleaned, cv.COLOR_BGR2RGB)  # Convertir en RGB avant d'afficher
    inpainted_image,dca_mask = inpainting_dca(image_cleaned_rgb)
    five_masks = five_segmentation(inpainted_image)    
    five_masks_cleaned = remove_small_parts_and_fill(five_masks)
    united_mask = union_mask(five_masks_cleaned)
    united_mask = np.array(united_mask, dtype=np.uint8)
    if dca_mask is not None:
        final_mask = postprocessing(united_mask, dca_mask)
    else:
        final_mask = united_mask
    final_mask_cleaned = remove_small_parts_and_fill([final_mask])[0]
    return final_mask_cleaned

def dice_score(mask1, mask2):
    intersection = np.sum(mask1 * mask2)
    union = np.sum(mask1) + np.sum(mask2)
    result = 2 * intersection / union  
    return result

# Création du set de données
def create_dataset(images_path, masks, output_dir):
    # Créer le répertoire de sortie s'il n'existe pas
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Parcourir les images et les masques
    for i, (image_path, mask) in enumerate(zip(images_path, masks)):
        mask = mask.astype(np.uint8)
        print(mask.shape)
        # Sauvegarder l'image et le masque final
        mask_name = os.path.basename(image_path) + '_pred_mask.png'
        io.imsave(os.path.join(output_dir, mask_name),mask )
        # Afficher la progression
        print(f'Processed {i+1}/{len(images_path)} images')


def resize_with_padding_binary_mask(mask, target_size):
    # Récupérer les dimensions du masque
    height, width = mask.shape[:2]

    # Calculer le ratio de redimensionnement en conservant la proportion
    if height > width:
        ratio = target_size[0] / height
    else:
        ratio = target_size[1] / width

    # Redimensionner le masque en conservant la proportion avec la méthode INTER_NEAREST
    resized_mask = cv.resize(mask, None, fx=ratio, fy=ratio, interpolation=cv.INTER_NEAREST)

    # Appliquer le padding pour obtenir une taille de 256x256
    pad_width = (target_size[1] - resized_mask.shape[1]) // 2
    pad_height = (target_size[0] - resized_mask.shape[0]) // 2
    padded_mask = cv.copyMakeBorder(resized_mask, pad_height, pad_height, pad_width, pad_width, cv.BORDER_CONSTANT, value=0)

    if padded_mask.shape[0] != target_size[0] or padded_mask.shape[1] != target_size[1]:
        padded_mask = cv.resize(padded_mask, target_size, interpolation=cv.INTER_NEAREST)
    return padded_mask


def compute_and_save_segmented_lesions(liste_chemins_images, output_dir):
    masks_pred_resized = []
    segmented_lesions = []
    segmented_lesions_square = []

    # Utilisation de tqdm pour afficher une barre de progression
    for i in tqdm(range(0, len(liste_chemins_images))):
        image = io.imread(liste_chemins_images[i])
        mask_pred = compute_segmentation(liste_chemins_images[i])
        #normaliser les masks
        mask_pred_normalized = np.array(mask_pred.astype(float) / mask_pred.max()).astype(int)
        
        mask_pred = mask_pred_normalized
        lesions_r = image[:,:,0] * mask_pred
        lesions_g = image[:,:,1] * mask_pred
        lesions_b = image[:,:,2] * mask_pred
        lesions = np.stack([lesions_r, lesions_g, lesions_b], axis=2)
        resized_padded_image = resize_with_padding_binary_mask(lesions, (256, 256))
        segmented_lesions.append(lesions)
        masks_pred_resized.append(mask_pred_normalized)
        segmented_lesions_square.append(resized_padded_image)
    create_dataset(liste_chemins_images, segmented_lesions_square, output_dir)
    