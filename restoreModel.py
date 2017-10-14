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
model_path = '/Users/jesuspereyra/Desktop/modelWB'
saver = tf.train.import_meta_graph('/Users/jesuspereyra/Desktop/modelWB/modelWB.meta')
saver.restore(session, '/Users/jesuspereyra/Desktop/modelWB/modelWB')

graph = tf.get_default_graph()
final_tensor = graph.get_tensor_by_name("final_result:0")
start_times = time.time()

def test(image1):
  image1 = image1.reshape(conf.img_size_flat)
  feed_dict = {"x:0": [image1]}
  values = session.run(final_tensor, feed_dict=feed_dict)
  maxValue = session.run(tf.argmax(values, dimension=1))
  percentege = session.run(final_tensor, feed_dict=feed_dict)
  print(percentege)
  print('\nPrediction: ', conf.classes[maxValue[0]])
  return(conf.classes[maxValue[0]])