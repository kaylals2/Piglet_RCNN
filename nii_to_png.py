
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import cv2
import shutil

# Helper function: nii to png in x direction
# Iterate through .nii slices in x direction, save each slice as an individual .png
def nii_to_img(src,dst,cmap='viridis'):
    img = nib.load(src)
    data = img.get_fdata()
    print(data.shape)
    for x in range(data.shape[0]): #iterating over x
        slice_data = data[x,:,:]
        plt.imsave(dst + '_slice_' + str(x) + '.png', slice_data, cmap=cmap)


image_name = 'oT1.nii'
annotation_name = 'manual_brain_mask-label-1-resampled.nii'
base_folder = 'C:\\Users\\Kayla\\Desktop\\scripts\\manual_extraction_files'

# Empty images and annotations before populating
shutil.rmtree('images')
os.mkdir('images')
shutil.rmtree('annotations')
os.mkdir('annotations')

# Populate annotation folder with .pngs
for folder in os.listdir(base_folder): # list all files/dirs in folder
    folder_path = os.path.join(base_folder, folder) # join arguments with os specific seperator.
    if os.path.isdir(folder_path): # only select directories.
        print(folder)
        image_path = os.path.join(folder_path, image_name)
        nii_to_img(image_path, os.path.join('images', folder))
        annotation_path = os.path.join(folder_path, annotation_name)
        nii_to_img(annotation_path, os.path.join('annotations', folder), cmap='gray')

        
