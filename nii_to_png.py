
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import cv2
import shutil

#helper function nii to png in x direction
def nii_to_img(src,dst,cmap='viridis'):
    img = nib.load(src)
    data = img.get_fdata()
    print(data.shape)
    for x in range(data.shape[0]): # assume iterating over x
        slice_data = data[x,:,:]
        plt.imsave(dst + '_slice_' + str(x) + '.png', slice_data, cmap=cmap)


image_name = 'oT1.nii'
# annotation_name = 'manual_brain_mask-label.nii'
annotation_name = 'manual_brain_mask-label-resampled.nii'
base_folder = 'C:\\Users\\Kayla\\Desktop\\scripts\\manual_extraction_files'

# empty images and annotations before populating
shutil.rmtree('images')
os.mkdir('images')
shutil.rmtree('annotations')
os.mkdir('annotations')

for folder in os.listdir(base_folder): # list all files/dirs in folder
    folder_path = os.path.join(base_folder, folder) # join arguments with os specific seperator.
    if os.path.isdir(folder_path): # only select directories.
        print(folder)
        image_path = os.path.join(folder_path, image_name)
        nii_to_img(image_path, os.path.join('images', folder))
        annotation_path = os.path.join(folder_path, annotation_name)
        nii_to_img(annotation_path, os.path.join('annotations', folder),cmap='gray')




# file_path = 'C:\\Users\\Kayla\\Desktop\\scripts\\manual_extraction_files\\Myelin1_01\\oT1.nii'
# img = nib.load(file_path)
# data = img.get_fdata()
# print(img.get_data_dtype())
# print(data)
# print(data.shape)
# slices = []

# for x in range(data.shape[0]): # assuming (x,y,z)
#     slice_data = data[x,:,:]
#     'x_' + str(x) + '.png' # x is index value
#     # save image
#     plt.imsave(img_path, slice_data, cmap='gray')
#     # show image
#     plt.imshow(slice_data, cmap='gray')
#     plt.show()


# for y in range(data.shape[1]): #assuming (x,y,z)
#     plt.imsave('y_' + str(y) + '.png', data[:,y,:], cmap='gray')
# for z in range(data.shape[2]): #assuming (x,y,z)
#     plt.imsave('z_' + str(z) + '.png', data[:,:,z], cmap='gray')


