import numpy as np #'Numerical Python' package
import nibabel as nib #Neuroimaging file format package 
import io
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import tensorflow as tf
import os

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_utils


def load_images_from_nii(path, cmap='viridis', start_from=0): #max256
    res = []
    nii = nib.load(path)
    data = nii.get_fdata()
    print(data)
    print(data.shape) # (256, 288, 288)
    for x in range(start_from,data.shape[0]):
        print(f"Loading {x}/{data.shape[0]}")
        slice_data = data[x,:,:]
        buf = io.BytesIO() # Create in memory file
        plt.imsave(buf, slice_data, cmap=cmap) # Save to inmem file
        buf.seek(0) # go to beginning of inmem file
        img = Image.open(buf).convert('RGB')
        (im_width, im_height) = img.size
        img = np.array(img.getdata()).reshape(
            (1, im_height, im_width, 3)).astype(np.uint8)
        res.append(img)
        buf.close()
    return res

# Take affine from input nii file
def single_nii(nii_path, detection_model):
    nii = nib.load(nii_path)
    affine = nii.affine
    images = load_images_from_nii(nii_path)

# Detect for each
    detection_masks_l = np.zeros(nii.shape) # detection mask list to avoid naming conflict
    for i, image in enumerate(images):
        print(f"Detecting {i}/{len(images)}")
        detections = detection_model(image)

        # Tensor to numpy array
        detections = {key:value.numpy() for key, value in detections.items()}
        image_with_detections = image.copy()

# Convert np.arrays to tensors
        if 'detection_masks' in detections:
            detection_masks = tf.convert_to_tensor(detections['detection_masks'][0])
            detection_boxes = tf.convert_to_tensor(detections['detection_boxes'][0])

            #  Reframe the the bbox mask to the image size.
            confidence = 0.5
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes,
                        image.shape[1], image.shape[2])
            detection_masks_reframed = tf.cast(detection_masks_reframed > confidence,
                                                tf.uint8)
            detection_masks_reframed_np = detection_masks_reframed.numpy()
            detections['detection_masks_reframed'] = detection_masks_reframed_np
            # print(detection_masks_reframed_np[0])
            # print(detection_masks_reframed_np[0].shape)
            detection_masks_l[i] = detection_masks_reframed_np[0] #idk take the first mask? or mask with id 0?

# https://nipy.org/nibabel/nifti_images.html
    # print(detection_masks_l.shape)
    out_nii = nib.nifti1.Nifti1Image(detection_masks_l.astype(np.uint8), affine)
    # print(out_nii.shape)
    nib.save(out_nii, os.path.join(os.path.dirname(nii_path), 'auto_Model_B.nii'))

SINGLE_NII_PATH = './manual_extraction_files/Myelin1_01/oT1.nii'
MULTI_NII_PATH = './manual_extraction_files'

#EVAL_FOLDERS = ['Myelin1_09', 'Myelin1_19', 'Myelin1_20', 'Myelin1_28', 'Myelin1_31']
#EVAL_FOLDERS = ['Myelin1_10','Myelin1_17','Myelin1_26','Myelin1_37','Myelin1_44']

MODEL_PATH = './inference_graph/saved_model'
LABEL_PATH = './label_map.txt'
INPUT_NAME = 'oT1.nii'

# Load model and label map
detection_model = tf.saved_model.load(MODEL_PATH)
# category_index = label_map_util.create_category_index_from_labelmap(LABEL_PATH, use_display_name=True)

for folder in os.listdir(MULTI_NII_PATH): # list all files/dirs in folder
    folder_path = os.path.join(MULTI_NII_PATH, folder) # join arguments with os specific seperator.
    #if os.path.isdir(folder_path) and folder in EVAL_FOLDERS: # only select directories. 
    print(folder)
    nii_path = os.path.join(folder_path, INPUT_NAME)
    single_nii(nii_path, detection_model)

#single_nii(SINGLE_NII_PATH, detection_model)

