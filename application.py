import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# What our accuracy cutoff is
THRESH = 0.5

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
# PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
PATH_TO_LABELS = (os.path.join('/mnt/c/Users/Kevin/Desktop/kino_project/models/research/object_detection/data', 'mscoco_label_map.pbtxt'))



# opener = urllib.request.URLopener()
# opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
# tar_file = tarfile.open(MODEL_FILE)
# for file in tar_file.getmembers():
#   file_name = os.path.basename(file.name)
#   if 'frozen_inference_graph.pb' in file_name:
#     tar_file.extract(file, os.getcwd())


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

def load_image_into_numpy_array(image, im_width, im_height):
  # FOR FEEDING IMAGES
  # (im_width, im_height) = image.size
  # return np.array(image.getdata()).reshape(
  #     (im_height, im_width, 3)).astype(np.uint8)

  return image.reshape((im_height, im_width, 3)).astype(np.uint8)

# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
# PATH_TO_TEST_IMAGES_DIR = '/mnt/c/Users/Kevin/Desktop/kino_project/models/research/object_detection/test_images'
# TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

# # Size, in inches, of the output images.
# IMAGE_SIZE = (12, 8)

def run_inference_for_single_image(image, graph):
  with graph.as_default():
  
    # Get handles to input and output tensors
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in [
        'num_detections', 'detection_boxes', 'detection_scores',
        'detection_classes', 'detection_masks'
    ]:
      tensor_name = key + ':0'
      if tensor_name in all_tensor_names:
        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
            tensor_name)
    if 'detection_masks' in tensor_dict:
      # The following processing is only for single image
      detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
      detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
      # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
      real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
      detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
      detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
      detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
          detection_masks, detection_boxes, image.shape[0], image.shape[1])
      detection_masks_reframed = tf.cast(
          tf.greater(detection_masks_reframed, 0.5), tf.uint8)
      # Follow the convention by adding back the batch dimension
      tensor_dict['detection_masks'] = tf.expand_dims(
          detection_masks_reframed, 0)
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    # Run inference
    output_dict = tf.Session().run(tensor_dict,
                           feed_dict={image_tensor: np.expand_dims(image, 0)})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
      output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


def get_results(image_np, width, height):
  # image_np = load_image_into_numpy_array(image, width, height)
  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
  image_np_expanded = np.expand_dims(image_np, axis=0)
  # Actual detection.
  output_dict = run_inference_for_single_image(image_np, detection_graph)

  results = []
  for i in range(len(output_dict['detection_scores'])):
    score = output_dict['detection_scores'][i]
    location = output_dict['detection_boxes'][i]
    item = category_index[output_dict['detection_classes'][i]]['name']
    if (score > THRESH):
      results.append((score, item, location))
    else:
      break

  return results



# overall_results = {}

# for image_path in TEST_IMAGE_PATHS:
#   image = Image.open(image_path)
#   # the array based representation of the image will be used later in order to prepare the
#   # result image with boxes and labels on it.
#   image_np = load_image_into_numpy_array(image)
#   # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
#   image_np_expanded = np.expand_dims(image_np, axis=0)
#   # Actual detection.
#   output_dict = run_inference_for_single_image(image_np, detection_graph)
#   # Visualization of the results of a detection.
#   # vis_util.visualize_boxes_and_labels_on_image_array(
#   #     image_np,
#   #     output_dict['detection_boxes'],
#   #     output_dict['detection_classes'],
#   #     output_dict['detection_scores'],
#   #     category_index,
#   #     instance_masks=output_dict.get('detection_masks'),
#   #     use_normalized_coordinates=True,
#   #     line_thickness=8)
#   # plt.figure(figsize=IMAGE_SIZE)
#   # plt.imshow(image_np)
#   # plt.show()

#   # What do these things do?
#   # Detection boxes = give the coordinates for the box?
#   #     Given as an array of four values . . . 
#   # Detection classes = gives the index of the detected
#   # object in a given detection box (same shape as boxes)
#   #     Look at category index for actual object
#   # Detection scores = gives certainty of object detection
#   # in a given detection box
  
#   results = []
#   for i in range(len(output_dict['detection_scores'])):
#   	score = output_dict['detection_scores'][i]
#   	item = category_index[output_dict['detection_classes'][i]]['name']
#   	if (score > THRESH):
#   		results.append((score, item))
#   	else:
#   		break

#   overall_results[image_path.split("/")[-1]] = results

# print(overall_results)