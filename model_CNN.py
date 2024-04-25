import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import io
import cv2 as cv
import os
import DarkArtefactRemoval as dca
import dullrazor as dr
import segmentation_and_preprocessing as sp
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

######################################################
class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = self.X[idx]
        label = self.y[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Définir les transformations à appliquer aux images d'entraînement
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

def cnn_accuracy(predict,labels):
  accuracy = (predict == labels).sum()/(labels.shape[0])
  return accuracy

def vector_to_class(x):
  y = torch.argmax(nn.Softmax(dim=1)(x),axis=1)
  return y

######################################################
# Répertoire contenant les images
image_dir_train = 'Train/Train/'
image_dir_test = 'Test/Test/'

Train_path = image_dir_train

import glob
images_train = glob.glob(Train_path + '/*[0-9].jpg')
mask_img_train = glob.glob(Train_path + '/*seg.png')

images_with_mask = [ Train_path + mask_img_train[i].split('/')[-1].split('_seg')[0] + '.jpg' for i in range(len(mask_img_train))]
images_test = glob.glob(image_dir_test + '/*[0-9].jpg')
mask_img_test = glob.glob(image_dir_test + '/*seg.png')

#Lire le csv metadataTrain et metadataTest
metadataTrain = pd.read_csv('metadataTrain.csv')
metadataTest = pd.read_csv('metadataTest.csv')



# Checking if the number of images is right 
print('There are', len(images_train),  'train images')
print('There are', len(images_with_mask),  'train images with mask')
print('There are', len(mask_img_train),  'train masks')
print('There are', len(images_test),  'test images')
print('There are', len(mask_img_test),  'test masks')



X_train = glob.glob('output_masks_train_set_1/*.png')
X_train_names = [os.path.basename(x).split('.jpg_pred_mask.png')[0] for x in X_train]

# Get the names and classes as pandas Series
names_series = metadataTrain["ID"].loc[metadataTrain["ID"].isin(X_train_names)]
classes_series = metadataTrain["CLASS"].loc[metadataTrain["ID"].isin(X_train_names)]

# Convert the pandas Series to lists
names_list = names_series.tolist()
classes_list = classes_series.tolist()

X_train_ordered = []
X_train_ordered_names = []
for i in range(0, len(names_list)):
    for j in range(0, len(X_train_names)):
        if names_list[i] in X_train_names[j]:
            X_train_ordered.append(X_train[j])
            X_train_ordered_names.append(X_train_names[j])


X_train = [io.imread(x) for x in X_train_ordered]
X_train = np.array(X_train)
y_train = classes_list
y_train = np.array(y_train)
#----------------------------------
y_train = y_train - 1
#----------------------------------
print(X_train.shape)
print(y_train.shape)

#Créer la validation set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=50)




# Créer le jeu de données de segmentation
train_dataset = SegmentationDataset(X_train, y_train, transform=transform)



learning_rate = 0.01
n_epochs = 25
batch_size = 256
nb_classes = 8

nb_filters = 32         # number of convolutional filters to use
kernel_size = (3, 3)    # convolution kernel size
pool_size = (2, 2)      # size of pooling area for max pooling

# --- Size of the successive layers
n_h_0 = nb_channels = 3 #3 channels for RGB
n_h_1 = nb_filters
n_h_2 = nb_filters
n_h_3 = nb_filters
n_h_4 = nb_filters
n_h_5 = nb_filters
n_h_6 = nb_filters


model = torch.nn.Sequential(nn.Conv2d(in_channels=n_h_0, out_channels=n_h_1, kernel_size=kernel_size, stride=(1, 1), padding='same'),
                            nn.ReLU(),
                            nn.Conv2d(in_channels=n_h_1, out_channels=n_h_2, kernel_size=kernel_size, stride=(1, 1), padding='same'),
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size = pool_size, stride= (2,2)),
                            nn.Conv2d(in_channels=n_h_2, out_channels=n_h_3, kernel_size=kernel_size, stride=(1, 1), padding='same'),
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size = pool_size, stride= (2,2)),
                            nn.Conv2d(in_channels=n_h_3, out_channels=n_h_4, kernel_size=kernel_size, stride=(1, 1), padding='same'),
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size = pool_size, stride= (2,2)),
                            nn.Conv2d(in_channels=n_h_4, out_channels=n_h_5, kernel_size=kernel_size, stride=(1, 1), padding='same'),
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size = pool_size, stride= (2,2)),
                            nn.Conv2d(in_channels=n_h_5, out_channels=n_h_6, kernel_size=kernel_size, stride=(1, 1), padding='same'),
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size = pool_size, stride= (2,2)),
                            
                            nn.Flatten(),
                            nn.Linear(in_features = int(n_h_6 * 8* 8) , out_features = nb_classes))



train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu")
print('device',device)
model = model.to(device)



train_losses=[]
valid_losses=[]

for epoch in range(0,n_epochs):
  train_loss=0.0
  all_labels = []
  all_predicted = []

  for batch_idx, (imgs, labels) in enumerate(tqdm(train_loader)):

    # pass the samples through the network
    predict = model.forward(imgs.to(device))
    # apply loss function
    loss = criterion(predict, labels.to(device))
    # set the gradients back to 0
    optimizer.zero_grad()
    # backpropagation
    loss.backward()
    # parameter update
    optimizer.step()
    # compute the train loss
    train_loss += loss.item()
    # store labels and class predictions
    all_labels.extend(labels.tolist())
    all_predicted.extend(vector_to_class(predict).tolist())
    # Enregistrer le modèle à chaque époque
    if epoch % 1 == 0:
        torch.save(model.state_dict(), 'model_save/modele_epoch_{}.pth'.format(epoch))

  print('Epoch:{} Train Loss:{:.4f}'.format(epoch,train_loss/len(train_loader.dataset)))

  # calculate accuracy
  print('Accuracy:{:.4f}'.format(cnn_accuracy(np.array(all_predicted),np.array(all_labels))))

