import os
import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
import matplotlib.pyplot as plt
import cv2 as cv
import io
import tensorflow as tf
from object_detection.utils import dataset_util
import random
import PIL


def get_bounding_boxes(annotation):
  img = nib.load(annotation) 
  data = img.get_fdata() # returns numpy array with ? loaded in it
  res = []
  #for y in range(data.shape[1]):
  for x in range(data.shape[0]): #(x,y,z)
    slice_data = data[x,:,:]
    #slice_data = data[:,y,:]
    buf = io.BytesIO()
    plt.imsave(buf, slice_data, cmap='gray')
    buf.seek(0)
    # img = cv.imread(buf)
    img = cv.imdecode(np.frombuffer(buf.read(), dtype=np.uint8), cv.IMREAD_GRAYSCALE)
    contours, hierarchy = cv.findContours(img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    if len(contours) > 0:
      cnt = contours[0]
      # for cnt in contours: # for multiple bounding boxes
      x,y,w,h = cv.boundingRect(cnt)
      xmins.append(x / 288)
      xmaxs.append((x+w) / 288)
      ymins.append(y / 288)
      ymaxs.append((y+h) / 288)
    res.append((xmins, xmaxs, ymins, ymaxs))
  return res

def create_tf_example(example):
  height = 288 # Image height
  width = 288 # Image width
  filename = example['filename'].encode('utf-8') # Filename of the image. Empty if image is not from file
  pil_image = PIL.Image.open(example['filename'])
  output_io = io.BytesIO()
  pil_image.save(output_io, format='PNG')
  encoded_image_data = output_io.getvalue()
  image_format = b'png' # b'jpeg' or b'png'

  xmins = example['xmins'] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = example['xmaxs'] # List of normalized right x coordinates in bounding box
             # (1 per box)
  ymins = example['ymins'] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = example['ymaxs'] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
  classes_text = ['Brain'.encode('utf-8')] # List of string class name of bounding box (1 per box)
  classes = [1] # List of integer class id of bounding box (1 per box)

  # with tf.io.gfile.GFile(example['mask'], 'rb') as fid: # opens filename and set mask
  #   mask = fid.read()
  pil_image = PIL.Image.open(example['mask'])
  output_io = io.BytesIO()
  pil_image.convert('L').save(output_io, format='PNG')
  mask = output_io.getvalue()
  masks = [mask]

  # dictionary describing the features
  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/mask': dataset_util.bytes_list_feature(masks)
  }))
  return tf_example

def main():
    test_writer = tf.io.TFRecordWriter('test.record')
    train_writer = tf.io.TFRecordWriter('train.record')
    #annotation_name = 'manual_brain_mask-label-1.nii'
    annotation_name = 'manual_brain_mask-label-resampled.nii'
    #annotation_name = 'manual_brain_mask-label-1-resampled.nii'
    base_folder = 'C:\\Users\\Kayla\\Desktop\\scripts\\manual_extraction_files'

    for folder in os.listdir(base_folder): # list all files/dirs in folder
        folder_path = os.path.join(base_folder, folder) # join arguments with os specific seperator.
        if os.path.isdir(folder_path): # only select directories.
            annotation_path = os.path.join(folder_path, annotation_name)
            # print(annotation_path)
            # if random.randint(1,101) > 85:
            if folder in ['Myelin1_09','Myelin1_19','Myelin1_20','Myelin1_28','Myelin1_31']: #Split A
            #if folder in ['Myelin1_10,','Myelin1_10','Myelin1_17','Myelin1_26','Myelin1_37','Myelin1_44']: #Split B
              writer = test_writer
              print('test', folder)
            else:
              writer = train_writer
              print('train', folder)
            # get_bounding_boxes returns a list of bounding boxes per slice
            for i, (xmins, xmaxs, ymins, ymaxs) in enumerate(get_bounding_boxes(annotation_path)):
                # print(xmins, xmaxs, ymins, ymaxs)
                #creating an object (with curly brackets)
                example = {
                    'filename': os.path.join('images', folder) + '_slice_' + str(i) + '.png',
                    'mask': os.path.join('annotations', folder) + '_slice_' + str(i) + '.png',
                    'xmins': xmins,
                    'xmaxs': xmaxs,
                    'ymins': ymins,
                    'ymaxs': ymaxs
                }
                # print(example)
                tf_example = create_tf_example(example)
                #if len(xmins) > 0: ##remove empty slices from tf record
                # if random.randint(1,101) > 80:
                #   test_writer.write(tf_example.SerializeToString())
                # else:
                #   train_writer.write(tf_example.SerializeToString())
                writer.write(tf_example.SerializeToString())
                
    test_writer.close()
    train_writer.close()


if __name__ == '__main__':
    main()

#inspect the tf.record file
#for example in tf.python_io.tf_record_iterator("example.record"):
 #   print(tf.train.Example.FromString(example))



