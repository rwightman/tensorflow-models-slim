# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Provides utilities to preprocess images for the Inception networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.ops import control_flow_ops
from preprocessing.inception_preprocessing import \
  apply_with_random_selector, distort_color, distorted_bounding_box_crop

IMAGENET_MEAN_CAFFE = [123.68, 116.779, 103.939]  # RGB
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
VAR_EPS = 1e-3


def image_normalize(
        image,
        method='default',
        dataset_mean=IMAGENET_MEAN,
        dataset_std=IMAGENET_STD):
  """
  Args:
      image: Input image
      method: Normalization method
      dataset_mean: Mean dataset image value for normalization
      dataset_std: Std deviation of dataset image values

  Returns:

  """
  if method == 'caffe' or method == 'caffe_bgr':
    print('Caffe BGR normalize', image.get_shape())
    # Rescale to [0, 255]
    image = tf.multiply(image, 255.0)
    tf.subtract(image, IMAGENET_MEAN_CAFFE)
    # Convert RGB to BGR
    red, green, blue = tf.split(2, 3, image)
    image = tf.concat(2, [blue, green, red])
  elif method == 'caffe_rgb':
    print('Caffe RGB normalize', image.get_shape())
    # Rescale to [0, 255]
    image = tf.multiply(image, 255.0)
    image = tf.subtract(image, IMAGENET_MEAN_CAFFE)
  elif method == 'frame':
    print("Per-frame standardize", image.get_shape())
    mean, var = tf.nn.moments(image, axes=[0, 1], shift=0.3)
    std = tf.sqrt(tf.add(var, VAR_EPS))
    image = tf.subtract(image, mean)
    image = tf.divide(image, std)
  elif method == 'dataset':
    print('Dataset standardize', image.get_shape())
    image = tf.subtract(image, dataset_mean)
    image = tf.divide(image, dataset_std)
  else:
    assert method == 'default' or method == 'inception'
    print('Inception normalize [-1, 1]', image.get_shape())
    # Rescale to [-1,1] instead of [0, 1)
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
  return image


def preprocess_for_train(
        image, height, width, bbox,
        do_distort_color=True,
        do_flip_horizontal=True,
        do_flip_vertical=False,
        area_range=(0.2, 1.0),
        aspect_ratio_range=(0.75, 1.333),
        normalize_method='inception',
        fast_mode=True,
        scope=None,
        add_image_summaries=True):
  """Distort one image for training a network.
  """
  with tf.name_scope(scope, 'distort_image', [image, height, width, bbox]):
    if bbox is None:
      bbox = tf.constant(
        [0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    if image.dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    if add_image_summaries:
      # Each bounding box has shape [1, num_boxes, box coords] and
      # the coordinates are ordered [ymin, xmin, ymax, xmax].
      image_with_box = tf.image.draw_bounding_boxes(
        tf.expand_dims(image, 0), bbox)
      tf.summary.image('image_with_bounding_boxes', image_with_box)

    distorted_image, distorted_bbox = distorted_bounding_box_crop(
      image,
      bbox,
      area_range=area_range,
      aspect_ratio_range=aspect_ratio_range)

    # Restore the shape since the dynamic slice based upon the bbox_size loses
    # the third dimension.
    distorted_image.set_shape([None, None, 3])

    if add_image_summaries:
      image_with_distorted_box = tf.image.draw_bounding_boxes(
        tf.expand_dims(image, 0), distorted_bbox)
      tf.summary.image(
        'images_with_distorted_bounding_box', image_with_distorted_box)

    # This resizing operation may distort the images because the aspect
    # ratio is not respected. We select a resize method in a round robin
    # fashion based on the thread number.
    # Note that ResizeMethod contains 4 enumerated resizing methods.

    # We select only 1 case for fast_mode bilinear.
    num_resize_cases = 1 if fast_mode else 4
    distorted_image = apply_with_random_selector(
        distorted_image,
        lambda x, method: tf.image.resize_images(x, [height, width], method),
        num_cases=num_resize_cases)

    if add_image_summaries:
      tf.summary.image(
        'cropped_resized_image', tf.expand_dims(distorted_image, 0))

    # Randomly flip the image horizontally.
    if do_flip_horizontal:
        distorted_image = tf.image.random_flip_left_right(distorted_image)

    # Randomly flip the image horizontally.
    if do_flip_vertical:
        distorted_image = tf.image.random_flip_up_down(distorted_image)

    if do_distort_color:
        # Randomly distort the colors. There are 4 ways to do it.
        distorted_image = apply_with_random_selector(
           distorted_image,
           lambda x, ordering: distort_color(x, ordering, fast_mode),
           num_cases=4)

    if add_image_summaries:
      tf.summary.image(
        'final_distorted_image', tf.expand_dims(distorted_image, 0))

    distorted_image = image_normalize(distorted_image, method=normalize_method)
    return distorted_image


def preprocess_for_eval(image, height, width,
                        central_fraction=0.875, scope=None):
  """Prepare one image for evaluation.

  If height and width are specified it would output an image with that size by
  applying resize_bilinear.

  If central_fraction is specified it would crop the central fraction of the
  input image.

  Args:
    image: 3-D Tensor of image. If dtype is tf.float32 then the range should be
      [0, 1], otherwise it would converted to tf.float32 assuming that the range
      is [0, MAX], where MAX is largest positive representable number for
      int(8/16/32) data type (see `tf.image.convert_image_dtype` for details).
    height: integer
    width: integer
    central_fraction: Optional Float, fraction of the image to crop.
    scope: Optional scope for name_scope.
  Returns:
    3-D float Tensor of prepared image.
  """
  with tf.name_scope(scope, 'eval_image', [image, height, width]):
    if image.dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # Crop the central region of the image with an area containing 87.5% of
    # the original image.
    if central_fraction:
      image = tf.image.central_crop(image, central_fraction=central_fraction)

    if height and width:
      # Resize the image to the specified height and width.
      image = tf.expand_dims(image, 0)
      image = tf.image.resize_bilinear(image, [height, width],
                                       align_corners=False)
      image = tf.squeeze(image, [0])
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image


def preprocess_image(
        image, height, width,
        is_training=False,
        bbox=None,
        fast_mode=True,
        add_image_summaries=True,
        **kwargs):
  """Pre-process one image for training or evaluation.
  """
  do_distort_color = kwargs.pop('do_distort_color', False)
  do_flip_horizontal = kwargs.pop('do_flip_horizontal', True)
  do_flip_vertical = kwargs.pop('do_flip_vertical', False)
  area_range = kwargs.pop('area_range', (0.9, 1.0))
  aspect_ratio_range = kwargs.pop('aspect_ratio_range', (0.9, 1.111))
  central_fraction = kwargs.pop('central_fraction', 0.95)

  if is_training:
    return preprocess_for_train(
      image, height, width, bbox,
      do_distort_color=do_distort_color,
      do_flip_horizontal=do_flip_horizontal,
      do_flip_vertical=do_flip_vertical,
      area_range=area_range,
      aspect_ratio_range=aspect_ratio_range,
      fast_mode=fast_mode,
      add_image_summaries=add_image_summaries)
  else:
    return preprocess_for_eval(
      image, height, width,
      central_fraction=central_fraction)
