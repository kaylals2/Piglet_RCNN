import os
import nibabel as nib
from nibabel.processing import resample_from_to


BASE_FOLDER = 'C:\\Users\\Kayla\\Desktop\\scripts\\manual_extraction_files'
INPUT_NAME = 'manual_brain_mask-label.nii'
TEMPLATE_NAME = 'oT1.nii'
APPEND_OUTPUT = '-resampled'


# Subject masks are mapped to voxel space of structural images (256, 288, 288), without interpolation
def resample(input_path, template_path):
    input_ = nib.load(input_path)
    template = nib.load(template_path)

    output = resample_from_to(input_, template, order=0)
    print(template.shape,input_.shape,output.shape)
    nib.save(output, os.path.splitext(input_path)[0] + APPEND_OUTPUT + '.nii')

for folder in os.listdir(BASE_FOLDER): # list all files/dirs in folder
    folder_path = os.path.join(BASE_FOLDER, folder) # join arguments with os specific seperator.
    if os.path.isdir(folder_path): # only select directories.
        print(folder)
        input_path = os.path.join(folder_path, INPUT_NAME)
        template_path = os.path.join(folder_path, TEMPLATE_NAME)
        resample(input_path, template_path)
        




# INPUT_PATH = 'C:\\Users\\Kayla\\Desktop\\scripts\\manual_extraction_files\\Myelin1_01\\manual_brain_mask-label-1.nii'
# ORIG_PATH = 'C:\\Users\\Kayla\\Desktop\\scripts\\manual_extraction_files\\Myelin1_01\\oT1.nii'
# OUTPUT_ID = '-resampled'

# input_ = nib.load(INPUT_PATH)
# orig = nib.load(ORIG_PATH)

# output = resample_from_to(input_, orig, order=0)
# print(orig.shape,input_.shape,output.shape)
# nib.save(output, os.path.splitext(INPUT_PATH)[0] + OUTPUT_ID + '.nii')


