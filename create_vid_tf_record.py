# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert raw VID dataset to TFRecord for object_detection.

Example usage:
    ./create_pascal_tf_record --data_dir=/data2/imagenet/VID \
        --output_path=/data2/imagenet/VID
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os

from lxml import etree
import PIL.Image
import tensorflow as tf

#from object_detection.utils import dataset_util
#from object_detection.utils import label_map_util
from utils import dataset_util
from utils import label_map_util


flags = tf.app.flags
flags.DEFINE_string('data_dir', '/data2/imagenet/VID/train/ILSVRC', 'Root directory to raw VID dataset.')
flags.DEFINE_string('set', 'train', 'Convert training set, validation set or '
                    'merged set.')
flags.DEFINE_string('annotations_dir', 'Annotations',
                    '(Relative) path to annotations directory.')
flags.DEFINE_string('output_path', '/data2/imagenet/VID/TF/train_tf', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', 'data/vid_local_map.pbtxt',
                    'Path to label map proto')
FLAGS = flags.FLAGS

SETS = ['train', 'val', 'test']


def dict_to_tf_example(data,
                       dataset_directory,
                       label_map_dict,
                       setname = 'train'):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    dataset_directory: Path to root directory holding PASCAL dataset
    label_map_dict: A map from string label names to integers ids.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).
    image_subdirectory: String specifying subdirectory within the
      PASCAL dataset directory holding the actual image data.

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """

  img_path = os.path.join(data['folder'], data['filename'])
  full_path = os.path.join(dataset_directory, 'Data/VID', setname, img_path+'.JPEG')
  with tf.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  key = hashlib.sha256(encoded_jpg).hexdigest()

  width = int(data['size']['width'])
  height = int(data['size']['height'])

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []
  trackid = []
  occluded =  []
  generated = []

  if 'object' in data:
      for obj in data['object']:
        xmin.append(float(obj['bndbox']['xmin']) / width)
        ymin.append(float(obj['bndbox']['ymin']) / height)
        xmax.append(float(obj['bndbox']['xmax']) / width)
        ymax.append(float(obj['bndbox']['ymax']) / height)
        classes_text.append(obj['name'].encode('utf8'))
        classes.append(label_map_dict[obj['name']])
        trackid.append(int(obj['trackid']))
        occluded.append(int(obj['occluded']))
        generated.append(int(obj['generated']))

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/class/trackid': dataset_util.int64_list_feature(trackid),
      'image/object/class/occluded': dataset_util.int64_list_feature(occluded),
      'image/object/class/generated': dataset_util.int64_list_feature(generated),
  }))
  return example


def main(_):
  if FLAGS.set not in SETS:
    raise ValueError('set must be in : {}'.format(SETS))

  data_dir = FLAGS.data_dir

  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
  print ("output_path is :")
  print(FLAGS.output_path)

  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

  logging.info('Reading from VID dataset.')
  examples_path = os.path.join(data_dir,'ImageSets', 'VID','list'
                               ,FLAGS.set + '_list.txt')
  annotations_dir = os.path.join(data_dir, FLAGS.annotations_dir, 'VID', FLAGS.set)
  examples_list = dataset_util.read_examples_list(examples_path)
  for idx, example in enumerate(examples_list):
    if idx % 100 == 0:
      logging.info('On image %d of %d', idx, len(examples_list))
    path = os.path.join(annotations_dir, example)
    with tf.gfile.GFile(path, 'r') as fid:
      xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

    tf_example = dict_to_tf_example(data, FLAGS.data_dir, label_map_dict,
                                    FLAGS.set)
    writer.write(tf_example.SerializeToString())

  writer.close()


if __name__ == '__main__':
  tf.app.run()
