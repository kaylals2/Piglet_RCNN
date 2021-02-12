import os
import nibabel as nib
from nibabel.processing import resample_from_to

#Subject masks are mapped to voxel space of structural images (256, 288, 288), without interpolation

BASE_FOLDER = 'C:\\Users\\Kayla\\Desktop\\scripts\\manual_extraction_files'
INPUT_NAME = 'manual_brain_mask-label.nii'
TEMPLATE_NAME = 'oT1.nii'
APPEND_OUTPUT = '-resampled'


# Function to apply affine of structural image to manual masks
def resample(input_path, template_path):
    input_ = nib.load(input_path)
    template = nib.load(template_path)

    output = resample_from_to(input_, template, order=0)
    print(template.shape,input_.shape,output.shape)
    nib.save(output, os.path.splitext(input_path)[0] + APPEND_OUTPUT + '.nii')
    
# Iterates through folders to apply function to each subject 
for folder in os.listdir(BASE_FOLDER): # list all files/directories in base folder
    folder_path = os.path.join(BASE_FOLDER, folder) # join arguments with os specific seperator.
    if os.path.isdir(folder_path): # only select directories.
        print(folder)
        input_path = os.path.join(folder_path, INPUT_NAME)
        template_path = os.path.join(folder_path, TEMPLATE_NAME)
        resample(input_path, template_path)
        







