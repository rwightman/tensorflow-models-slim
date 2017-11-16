# Copyright 2017 Ross Wightman. All Rights Reserved.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import os
import os.path
import re

IMG_EXTENSIONS = ['.png', '.jpg', '.jpeg']


def natural_key(string_):
  """See http://www.codinghorror.com/blog/archives/001018.html"""
  return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def find_images_and_targets(
        folder,
        types=IMG_EXTENSIONS,
        class_to_idx=None,
        has_background=False,
        leaf_name_only=True,
        sort=True):

  if class_to_idx is None:
    class_to_idx = dict()
    build_class_idx = True
  else:
    build_class_idx = False

  labels = []
  filenames = []
  for root, subdirs, files in os.walk(folder, topdown=False):
    rel_path = os.path.relpath(root, folder) if (root != folder) else ''
    label = os.path.basename(rel_path) if leaf_name_only else rel_path.replace(os.path.sep, '_')
    if build_class_idx and not subdirs:
      class_to_idx[label] = 0
    for f in files:
      base, ext = os.path.splitext(f)
      if ext.lower() in types:
        filenames.append(os.path.join(root, f))
        labels.append(label)

  if build_class_idx:
    classes = sorted(class_to_idx.keys(), key=natural_key)
    for i, c in enumerate(classes):
      class_to_idx[c] = i + 1 if has_background else i

  images_and_targets = zip(filenames, [class_to_idx[l] for l in labels])
  if sort:
    images_and_targets = sorted(images_and_targets, key=lambda k: natural_key(k[0]))
  if build_class_idx:
    return images_and_targets, classes, class_to_idx
  else:
    return images_and_targets


def _load_image(filename):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_jpeg(image_string, channels=3)
  image_decoded = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)
  return image_decoded


class DatasetImageFolder:
  """ Dataset for reading images organized in folders by class.

  This dataset uses the tf.data.Dataset iterators and directly loads images
  from files in folders instead of relying on TFRecords format.

  By default the dataset is setup to work out of the box with imagenet for
  TF models that have 1001 classes and a background class at 0.
  """

  def __init__(
          self, root,
          split='train',
          num_classes=1001,
          has_background=True,
          labels_file='./datasets/imagenet_lsvrc_2015_synsets.txt'):

    if labels_file and os.path.exists(labels_file):
      class_to_idx = {}
      classes = []
      with open(labels_file) as fp:
        for i, label in enumerate(map(str.strip, fp)):
          class_to_idx[label] = i + 1 if has_background else i
          classes.append(label)
      images_and_targets = find_images_and_targets(
        root, class_to_idx=class_to_idx)
    else:
      images_and_targets, classes, class_to_idx = find_images_and_targets(
        root, has_background=has_background)

    if len(images_and_targets) == 0:
      raise (RuntimeError(
        "Found 0 images in subfolders of: " + root + "\n"
        "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

    self.root = root
    self.split = split
    self.num_classes = num_classes
    images, targets = zip(*images_and_targets)
    self.images = images
    self.targets = targets
    self.num_samples = len(images)
    self.classes = classes
    self.class_to_idx = class_to_idx

  def get_iterator(
          self,
          process_fn=lambda x: x, shuffle=False,
          batch_size=32, epochs=-1, num_threads=4, num_pull=1):

    def _parse_data(filename, label):
      image_decoded = _load_image(filename)
      image_processed = process_fn(image_decoded)
      return image_processed, label

    images_arr = np.array(self.images)
    targets_arr = np.array(self.targets)
    if shuffle:
      p = np.random.permutation(len(images_arr))
      images_arr = images_arr[p]
      targets_arr = targets_arr[p]
    dataset = tf.data.Dataset.from_tensor_slices((images_arr, targets_arr))
    if shuffle:
      dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.map(_parse_data, num_parallel_calls=num_threads)
    dataset = dataset.prefetch((num_threads + 1) * batch_size * num_pull)
    dataset = dataset.repeat(epochs)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(num_pull)

    return dataset.make_one_shot_iterator()

  def get_inputs(
          self,
          process_fn=lambda x: x, shuffle=False,
          batch_size=32, epochs=-1, num_threads=4, num_pull=1):

    return self.get_iterator(
      process_fn, shuffle, batch_size, epochs, num_threads, num_pull).get_next()


def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
  """Gets a dataset capable of reading images in <class>/img folder structure.
  """
  return DatasetImageFolder(dataset_dir, split=split_name)
