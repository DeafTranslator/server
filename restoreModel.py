import tensorflow as tf
import time
from datetime import timedelta
import random
import os
import glob
import plot as pl
#=====================
import dataset
import config as conf
import matplotlib.pyplot as plt
import base64
import numpy as np
#=====================


session = tf.Session()
saver = tf.train.import_meta_graph('/Users/jesuspereyra/Documents/server/model/model_epoch101iteration1515_0.332208.meta')
saver.restore(session, '/Users/jesuspereyra/Documents/server/model/model_epoch101iteration1515_0.332208')

graph = tf.get_default_graph()
final_tensor = graph.get_tensor_by_name("final_result:0")
start_times = time.time()
      

def test2(image1):
  # image1 = tf.image.decode_png(data)
  # image1 = tf.image.decode_png(image1)
  # plt.imshow(image1)
  # tf.reshape(image1, conf.img_size_flat)
  # image1 = tf.reshape(conf.img_size_flat)
  feed_dict = {"x:0": [image1]}
  values = session.run(final_tensor, feed_dict=feed_dict)
  maxValue = session.run(tf.argmax(values, dimension=1))
  percentege = session.run(final_tensor, feed_dict=feed_dict)
  print(percentege)
  print('\nPrediction: ', conf.classes[maxValue[0]])
  # print('Real: ', test_ids[i].split('-', 1)[0])
  time_difs = end_times - start_times
  print("Time: " + str(round(time_difs)))
  pl.plot_image(image1)
  print('\n')


def test():
  test_images, test_ids = dataset.read_test_set(conf.test_path2, conf.img_size)

  # image1 = np.frombuffer(data, np.uint8)
  # image1 = image1.astype('float32')
  # image1 = image1 / 255
  start_times = time.time()
  image1 = test_images[0]
  image1 = image1.reshape(conf.img_size_flat)
  feed_dict = {"x:0": [image1]}
  values = session.run(final_tensor, feed_dict=feed_dict)
  maxValue = session.run(tf.argmax(values, dimension=1))
  percentege = session.run(final_tensor, feed_dict=feed_dict)
  print(percentege)
  print('\nPrediction: ', conf.classes[maxValue[0]])
  # print('Real: ', test_ids[i].split('-', 1)[0])
  end_times = time.time()  
  time_difs = end_times - start_times
  print("Time: " + str(round(time_difs)))
  return(conf.classes[maxValue[0]])

