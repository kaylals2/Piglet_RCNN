
#1. resample_nii.py
#2. nii_to_png.py
#3. gen_tf_record.py
#4. Train: $ python C:\Users\Kayla\Documents\TensorFlow\models\research\object_detection\model_main_tf2.py --pipeline_config_path=C:\\Users\\Kayla\\Desktop\\scripts\\my_models\\my_mask_rcnn_inception_resnet_v2\\pipeline.config --model_dir=C:\\Users\\Kayla\\Desktop\\scripts\\my_models\\my_mask_rcnn_inception_resnet_v2 --alsologtostderr
#5. Generate inference graph: $ python C:\Users\Kayla\Documents\TensorFlow\models\research\object_detection\exporter_main_v2.py --trained_checkpoint_dir=C:\\Users\\Kayla\\Desktop\\scripts\\my_models\\my_mask_rcnn_inception_resnet_v2 --output_directory=C:\\Users\\Kayla\\Desktop\\scripts\\inference_graph --pipeline_config_path=C:\\Users\\Kayla\\Desktop\\scripts\\my_models\\my_mask_rcnn_inception_resnet_v2\\pipeline.config
#6. Eval: $ python C:\Users\Kayla\Documents\TensorFlow\models\research\object_detection\model_main_tf2.py --pipeline_config_path=C:\\Users\\Kayla\\Desktop\\scripts\\my_models\\my_mask_rcnn_inception_resnet_v2\\pipeline.config --model_dir=C:\\Users\\Kayla\\Desktop\\scripts\\my_models\\my_mask_rcnn_inception_resnet_v2 --checkpoint_dir=C:\\Users\\Kayla\\Desktop\\scripts\\inference_graph\\checkpoint --alsologtostderr
#7. Generate .nii files using multi_nii.py

import os 
#import subprocess


os.system('python resample_nii.py')
os.system('python nii_to_png.py')
os.system('python gen_tf_record.py')

pipeline_config_path = 'C:\\Users\\Kayla\\Desktop\\scripts\\my_models\\my_mask_rcnn_inception_resnet_v2\\pipeline.config'
model_dir = 'C:\\Users\\Kayla\\Desktop\\scripts\\my_models\\my_mask_rcnn_inception_resnet_v2'
trained_checkpoint_dir = 'C:\\Users\\Kayla\\Desktop\\scripts\\my_models\\my_mask_rcnn_inception_resnet_v2'
output_directory = 'C:\\Users\\Kayla\\Desktop\\scripts\\inference_graph'
checkpoint_dir = 'C:\\Users\\Kayla\\Desktop\\scripts\\inference_graph\\checkpoint'

os.system(f'python C:\\Users\\Kayla\\Documents\\TensorFlow\\models\\research\\object_detection\\model_main_tf2.py --pipeline_config_path={pipeline_config_path} --model_dir={model_dir} --alsologtostderr')
os.system(f'python C:\\Users\\Kayla\\Documents\\TensorFlow\\models\\research\\object_detection\\exporter_main_v2.py --trained_checkpoint_dir={trained_checkpoint_dir}--output_directory={output_directory} --pipeline_config_path={pipeline_config_path}')
os.system(f'python C:\\Users\\Kayla\\Documents\\TensorFlow\\models\\research\\object_detection\\model_main_tf2.py --pipeline_config_path={pipeline_config_path} --model_dir={model_dir} --checkpoint_dir={checkpoint_dir} --alsologtostderr')

